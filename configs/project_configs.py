import os
from dataclasses import dataclass

@dataclass
class ProjectConfigs:
    # directories
    HOME_DIR: str = os.environ['HOME']
    SCRIPT_PATH: str = './TrainingScript.py'
    INTERPRETER_PATH: str = HOME_DIR + '/anaconda3/envs/env_name/bin/python'
    QUEUE_PATH: str = './queue'
    DATA_PATH: str = HOME_DIR + '/data_directory/numpy'
    EXPEREMENT_PATH: str = './script_experiment'
