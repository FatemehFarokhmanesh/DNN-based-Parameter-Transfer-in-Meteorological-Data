import random
from time import sleep
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from utils.automation.storage import MultiRunExperiment
from utils.automation.devices import set_free_devices_as_visible


EXPERIMENT_DESCRIPTION = 'Example experiment for demonstrating the use of automation utils'
EXPERIMENT_PATH = './script_experiment'


if __name__ == '__main__':
    # setup argument parser to enable or disable GPU training
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-enabled', action='store_true', dest='use_gpu')
    parser.add_argument('--gpu-disabled', action='store_false', dest='use_gpu')
    parser.set_defaults(use_gpu=False) # --> use CPU by default
    args = parser.parse_args()

    if args.use_gpu:
        # automatically select GPU device
        device_id = set_free_devices_as_visible(num_devices=1)
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
    model = nn.Conv2d(4, 4, (3, 3), padding=1)
    model = model.to(device) # bring model to device
    optimizer = Adam(model.parameters(), lr=1.e-4)
    scheduler = StepLR(optimizer, 2, gamma=0.1)
    loss_func = nn.L1Loss()

    # store initial checkpoint
    training_state = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    run.save_epoch_state(training_state, epoch=0)
    # generates checkpoint with predefined name 'epoch_0.pth' in checkpoint directory
    # Alternative: run.save_checkpoint(training_state, 'epoch_0.pth')

    # a summary writer might be useful, as well
    summary = run.get_tensorboard_summary()
    # --> summary file is stored in summary subfolder of the run directory

    # do some training
    for i in range(10):

        inputs = torch.randn(10, 4, 32, 64).to(device)
        targets = 4. * inputs + 8. + torch.randn(10, 4, 32, 64).to(device)

        predictions = model(inputs)
        loss = loss_func(predictions, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # update summary
        summary.add_scalar('loss', loss.item())

        # store checkpoint every few batches
        if (i + 1) % 4 == 0:
            run.save_epoch_state(training_state, epoch=(i + 1))

    # store final checkpoint
    run.save_checkpoint(training_state, 'final_state.pth')

    print('[INFO] Finished training. Sleeping for a while...')
    sleep(60)
    print('[INFO] Exiting.')