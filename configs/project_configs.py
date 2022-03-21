import os
from dataclasses import dataclass

@dataclass
class ProjectConfigs:
    # directories
    HOME_DIR: str = os.environ['HOME']
    SCRIPT_PATH: str = './TrainingScript.py'
    INTERPRETER_PATH: str = HOME_DIR + '/anaconda3/envs/wbd/bin/python'
    QUEUE_PATH: str = './queue'
    DATA_PATH: str = HOME_DIR + '/codes/numpy/32_64'
    EXPEREMENT_PATH: str = './script_experiment'
