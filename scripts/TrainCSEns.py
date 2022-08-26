import os
import random
from time import sleep
import argparse
import numpy as np

import torch
import torch.nn as nn
import dask

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs import ProjectConfigs
from utils.automation.storage import MultiRunExperiment
from utils.automation.devices import set_free_devices_as_visible
from interface.zarr.data_storage import Variable3d, CoordinateAxis, MultiVariableData2d
from interface.zarr.utils import MergedData

from utils import WelfordStatisticsTracker, ProgressBar
from networks import SimpleResnet, UNet


dask.config.set(scheduler='synchronous')
torch.set_num_threads(6)

EXPERIMENT_DESCRIPTION = 'Example experiment for demonstrating the use of automation utils'
EXPERIMENT_PATH = ProjectConfigs().EXPEREMENT_PATH

def main():
    # setup argument parser to enable or disable GPU training
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-enabled', action='store_true', dest='use_gpu')
    parser.add_argument('--gpu-disabled', action='store_false', dest='use_gpu')
    parser.set_defaults(use_gpu=True) # --> use CPU by default
    args = parser.parse_args()

    if args.use_gpu:
        # automatically select GPU device
        device_id = set_free_devices_as_visible(num_devices=1)
        device = torch.device('cuda:0')
    else:
        # use CPU
        device_id = 'cpu'
        device = torch.device(device_id)

    # setup experiment directory
    experiment = MultiRunExperiment(EXPERIMENT_PATH, description=None)
    # descriptions may only be given for empty experiment directories
    # --> set experiment description to None to avoid conflicts when opening the same experiment directory
    #     multiple times (eg. from within different processes)
    print(f'[INFO] Created experiment directory at {experiment.path}')

    # but descriptions may be added for each run without conflict problems
    run_description = {
        'subject': EXPERIMENT_DESCRIPTION, # remember subject of the run
        'script': __file__, # remember script file name
        'base_directory': EXPERIMENT_PATH, # remember where the data was stored
        'device_id': device_id, # remember device ID
    }
    # create run directory
    run = experiment.create_new_run(description=run_description)
    print(f'[INFO] Created run directory at location {run.path}')

    # runs may also store experiment parameters
    run_params = {
        'random_number': random.random(), # get some random number and store it
        'some_other_random_number': random.random()
    }
    # add parameters to run directory
    run.add_parameter_settings(run_params)
    
    input_variables = [Variable3d.TK, 
                        Variable3d.U, 
                        Variable3d.V,
                        Variable3d.W,
                        Variable3d.Z,
                        Variable3d.RH,
                        Variable3d.QH,
                        Variable3d.DBZ
                        ]
    output_variables = [Variable3d.QV]

    train_members = np.arange(0, 199) 
    test_members = np.arange(200, 204) 

    domain_selection = {
        
        CoordinateAxis.LEVEL: [0, 1, 2], # range 0 <= x < 12
        CoordinateAxis.LAT: np.arange(0, 352),  # range 0 <= x < 352
        CoordinateAxis.LON: np.arange(0, 250),  # range 0 <= x < 250
        CoordinateAxis.TIME: [3, 4, 5] # range 0 <= x < 6
    }

    base_path = None  # change here if needed

    def get_dataset(members, scales=None):
        selection = {
            CoordinateAxis.MEMBER: members,
            **domain_selection
        }
        if scales is not None:
            input_scales, target_scales = scales
        else:
            input_scales, target_scales = None, None
        input_data = MultiVariableData2d(
            input_variables, selection=selection, base_path=base_path, scales=input_scales)
        input_scales = input_data.get_scales()
        target_data = MultiVariableData2d(
            output_variables, selection=selection, base_path=base_path, scales=target_scales)
        target_scales = target_data.get_scales()
        data = MergedData(input_data, target_data)
        return data, (input_scales, target_scales)
    batch_size = 20
    training_data, scales = get_dataset(train_members)
    validation_data, _ = get_dataset(test_members, scales=scales)
    training_loader = DataLoader(training_data, batch_size=batch_size, num_workers=8)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, num_workers=8)
    in_channels, out_channels = len(input_variables), len(output_variables)
    model = SimpleResnet(in_channels, out_channels, 64, 2).to(device)
    loss_function = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, threshold=0.001)
    num_epochs = 5
    stats_tracker = WelfordStatisticsTracker()
    
    # store initial checkpoint
    training_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler,
    }
    run.save_epoch_state(training_state, epoch=0)
    # generates checkpoint with predefined name 'epoch_0.pth' in checkpoint directory
    # Alternative: run.save_checkpoint(training_state, 'epoch_0.pth')

    # a summary writer might be useful, as well
    summary = run.get_tensorboard_summary()
    # --> summary file is stored in summary subfolder of the run directory
    
    for e in range(num_epochs):
        # Training loop
        model.train()
        stats_tracker.reset()
        print('[INFO] Starting epoch {}. {} batches to train.'.format(e, len(training_loader)))
        pbar = ProgressBar(len(training_data))
        for i, batch in enumerate(training_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            print(inputs.size(), targets.size())
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            stats_tracker.update(loss, weight=len(batch))
            pbar.step(batch[0].shape[0])
        summary.add_scalar("Loss/train", stats_tracker.mean(), e)
        summary.flush()
        print('[INFO] Training loss after epoch {}: {}+-{}'.format(e, stats_tracker.mean(), stats_tracker.std()))
        # Validation loop
        model.eval()
        stats_tracker.reset()
        print('[INFO] Validating model state.')
        with torch.no_grad():
            pbar = ProgressBar(len(validation_data))
            for i, batch in enumerate(validation_loader):
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                loss = loss_function(predictions, targets).item()
                stats_tracker.update(loss, weight=len(batch))
                pbar.step(batch[0].shape[0])

        scheduler.step(loss)
        summary.add_scalar("Loss/test", stats_tracker.mean(), e)
        summary.flush()
        # store checkpoint every few batches
        training_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
        }
        if (e + 1) % 4 == 0:
            run.save_epoch_state(training_state, epoch=(e + 1))

    # store final checkpoint    
    run.save_checkpoint(training_state, 'final_state.pth')
    print('[INFO] Finished training. Sleeping for a while...')
    sleep(60)
    print('[INFO] Exiting.')
        
if __name__ == '__main__':
    main()