import os
import random
from time import sleep
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.automation.storage import MultiRunExperiment
from utils.automation.devices import set_free_devices_as_visible
from configs import ProjectConfigs
from torch.utils.data import DataLoader

from interface.numpy.datasets import WeatherBenchData, TimeVariateData, ConstantData
from interface.numpy.datastorage import WeatherBenchNPYStorage
from interface.numpy.transforms import LatitudeStandardScaling, GlobalStandardScaling
from utils import WelfordStatisticsTracker, ProgressBar
from networks import SimpleResnet, UNet

torch.set_num_threads(6)
path = ProjectConfigs().DATA_PATH

data_path = {
    't2m': os.path.join(path, 't2m'),
    'tcc': os.path.join(path, 'tcc'),
    'u10': os.path.join(path, 'u10'),
    'v10': os.path.join(path, 'v10'),
    'tp': os.path.join(path, 'tp'),
    'tisr': os.path.join(path, 'tisr'),
    'orography': os.path.join(path, 'orography')
}

def data_pattern(min_date, max_date):
    # Define data set pattern for training
    wbd = WeatherBenchData(min_date=min_date, max_date=max_date, except_on_changing_date_bounds=False)
    wbd.add_data_group(
        'input_data', [
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['t2m']),
                name='t2m'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['tcc']),
                name='tcc'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['u10']),
                name='u10'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['v10']),
                name='v10'
            ),
            
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['tp']),
                name='tp'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['tisr']),
                name='tisr'
            ),
            ConstantData(
                WeatherBenchNPYStorage(data_path['orography']),
                name='orography'
            )
        ]
    )
    wbd.add_data_group(
        'target_data', [
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['t2m']),
                name='t2m'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['tcc']),
                name='tcc'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['u10']),
                name='u10'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['v10']),
                name='v10'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['tp']),
                name='tp'
            ),
            TimeVariateData(
                WeatherBenchNPYStorage(data_path['tisr']),
                name='tisr'
            )
        ]
    )
    return wbd

batch_size = 11
fitting_batch_size = 24

print('[INFO] Building training data.')
training_data = data_pattern('1980-01-01-00', '2000-01-01-00')
transformer_t2m = GlobalStandardScaling()
transformer_tcc = GlobalStandardScaling()
transformer_u10 = GlobalStandardScaling()
transformer_v10 = GlobalStandardScaling()
transformer_tp = LatitudeStandardScaling()
transformer_tisr = GlobalStandardScaling()
transformer_orography = GlobalStandardScaling()
transformer_t2m.fit(training_data.data_groups['input_data'][0], batch_size=fitting_batch_size)
transformer_tcc.fit(training_data.data_groups['input_data'][1], batch_size=fitting_batch_size)
transformer_u10.fit(training_data.data_groups['input_data'][2], batch_size=fitting_batch_size)
transformer_v10.fit(training_data.data_groups['input_data'][3], batch_size=fitting_batch_size)
transformer_tp.fit(training_data.data_groups['input_data'][4], batch_size=fitting_batch_size)
transformer_tisr.fit(training_data.data_groups['input_data'][5], batch_size=fitting_batch_size)
transformer_orography.fit(training_data.data_groups['input_data'][6], batch_size=fitting_batch_size)


transformer_t2m.fit(training_data.data_groups['target_data'][0], batch_size=fitting_batch_size)
transformer_tcc.fit(training_data.data_groups['target_data'][1], batch_size=fitting_batch_size)
transformer_u10.fit(training_data.data_groups['target_data'][2], batch_size=fitting_batch_size)
transformer_v10.fit(training_data.data_groups['target_data'][3], batch_size=fitting_batch_size)
transformer_tp.fit(training_data.data_groups['target_data'][4], batch_size=fitting_batch_size)
transformer_tisr.fit(training_data.data_groups['target_data'][5], batch_size=fitting_batch_size)

training_loader = DataLoader(training_data, batch_size=batch_size, num_workers=4)

print('[INFO] Building validation data.')
validation_data = data_pattern('2000-01-01-01', '2003-01-01-01')
transformer_t2m.fit(validation_data.data_groups['input_data'][0], batch_size=fitting_batch_size)
transformer_tcc.fit(validation_data.data_groups['input_data'][1], batch_size=fitting_batch_size)
transformer_u10.fit(validation_data.data_groups['input_data'][2], batch_size=fitting_batch_size)
transformer_v10.fit(validation_data.data_groups['input_data'][3], batch_size=fitting_batch_size)
transformer_tp.fit(validation_data.data_groups['input_data'][4], batch_size=fitting_batch_size)
transformer_tisr.fit(validation_data.data_groups['input_data'][5], batch_size=fitting_batch_size)
transformer_orography.fit(validation_data.data_groups['input_data'][6], batch_size=fitting_batch_size)

transformer_t2m.fit(validation_data.data_groups['target_data'][0], batch_size=fitting_batch_size)
transformer_tcc.fit(validation_data.data_groups['target_data'][1], batch_size=fitting_batch_size)
transformer_u10.fit(validation_data.data_groups['target_data'][2], batch_size=fitting_batch_size)
transformer_v10.fit(validation_data.data_groups['target_data'][3], batch_size=fitting_batch_size)
transformer_tp.fit(validation_data.data_groups['target_data'][4], batch_size=fitting_batch_size)
transformer_tisr.fit(validation_data.data_groups['target_data'][5], batch_size=fitting_batch_size)

validation_loader = DataLoader(validation_data, batch_size=batch_size, num_workers=4)


EXPERIMENT_DESCRIPTION = 'Example experiment for demonstrating the use of automation utils'
EXPERIMENT_PATH = ProjectConfigs().EXPEREMENT_PATH


if __name__ == '__main__':
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

    # build a PyTorch training setup
    model = SimpleResnet(6, 1, 64, 2).to(device)

    print(model)
    loss_function = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, threshold=0.001)
    num_epochs = 100
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

    # do some training
    for e in range(num_epochs):
        # Training loop
        model.train()
        stats_tracker.reset()
        print('[INFO] Starting epoch {}. {} batches to train.'.format(e, len(training_loader)))
        pbar = ProgressBar(len(training_data))
        for i, batch in enumerate(training_loader):
            inputs, targets = batch
            inputs = list(inputs)
            targets = list(targets)

            inputs_trans = []
            inputs_trans.append(transformer_t2m.transform(inputs[0]))
            inputs_trans.append(transformer_tcc.transform(inputs[1]))
            inputs_trans.append(transformer_u10.transform(inputs[2]))
            inputs_trans.append(transformer_v10.transform(inputs[3]))
            inputs_trans.append(transformer_tp.transform(inputs[4]))
            inputs_trans.append(transformer_tisr.transform(inputs[5]))
            inputs_trans.append(transformer_orography.transform(inputs[6]))
            
            targets_trans = []
            targets_trans.append(transformer_t2m.transform(targets[0]))
            targets_trans.append(transformer_tcc.transform(targets[1]))
            targets_trans.append(transformer_u10.transform(targets[2]))
            targets_trans.append(transformer_v10.transform(targets[3]))
            targets_trans.append(transformer_tp.transform(targets[4]))
            targets_trans.append(transformer_tisr.transform(targets[5]))
            # to be modified based on the parameter confgurations
            # trained without t2m
            inputs_ts = inputs_trans[1:7]
            inputs_ts = torch.cat(inputs_ts, dim=1).to(dtype=torch.float32, device=device)
            inputs_ts = torch.squeeze(inputs_ts)

            targets_ts = targets_trans[0]
            targets_ts = torch.squeeze(targets_ts)
            targets_ts = torch.unsqueeze(targets_ts, 1).to(device)

            optimizer.zero_grad()
            predictions = model(inputs_ts)
            loss = loss_function(predictions, targets_ts)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            stats_tracker.update(loss, weight=len(batch))
            pbar.step(batch[0][0].shape[0])
            
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
                inputs = list(inputs)
                targets = list(targets)

                inputs_trans = []
                inputs_trans.append(transformer_t2m.transform(inputs[0]))
                inputs_trans.append(transformer_tcc.transform(inputs[1]))
                inputs_trans.append(transformer_u10.transform(inputs[2]))
                inputs_trans.append(transformer_v10.transform(inputs[3]))
                inputs_trans.append(transformer_tp.transform(inputs[4]))
                inputs_trans.append(transformer_tisr.transform(inputs[5]))
                inputs_trans.append(transformer_orography.transform(inputs[6]))
                
                targets_trans = []
                targets_trans.append(transformer_t2m.transform(targets[0]))
                targets_trans.append(transformer_tcc.transform(targets[1]))
                targets_trans.append(transformer_u10.transform(targets[2]))
                targets_trans.append(transformer_v10.transform(targets[3]))
                targets_trans.append(transformer_tp.transform(targets[4]))
                targets_trans.append(transformer_tisr.transform(targets[5]))
                
                inputs_ts = inputs_trans[1:7]
                inputs_ts = torch.cat(inputs_ts, dim=1).to(dtype=torch.float32, device=device)
                inputs_ts = torch.squeeze(inputs_ts)
                targets_ts = targets_trans[0]
                targets_ts = torch.squeeze(targets_ts)
                targets_ts = torch.unsqueeze(targets_ts, 1).to(device)

                predictions = model(inputs_ts)
                loss = loss_function(predictions, targets_ts).item()
                stats_tracker.update(loss, weight=len(batch))
                pbar.step(batch[0][0].shape[0])
        scheduler.step(loss)
        summary.add_scalar("Loss/test", stats_tracker.mean(), e)
        summary.flush()
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
