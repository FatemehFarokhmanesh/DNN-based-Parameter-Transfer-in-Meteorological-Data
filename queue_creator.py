from utils.automation.queue import QueueManager, JobMaker
from configs import ProjectConfigs

# indicate script path
script_path = ProjectConfigs().SCRIPT_PATH

# indicate interpreter path
interpreter_path = ProjectConfigs().INTERPRETER_PATH # adapt that for your purposes

# indicate where to setup the queue
queue_path = ProjectConfigs().QUEUE_PATH


if __name__ == '__main__':
    # set up queue and get QueueManager
    queue = QueueManager(queue_path, make_directories=True)
    
    # get job makers that return a single run configuration
    job_maker_gpu = JobMaker(interpreter_path, script_path)

    # push job to the queue
    job_maker_gpu.export_jobs(queue)
    
    print('[INFO] Pushed jobs to queue')
