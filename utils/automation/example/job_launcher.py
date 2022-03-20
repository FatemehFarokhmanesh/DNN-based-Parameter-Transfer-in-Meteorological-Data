from utils.automation.queue import QueueManager, JobLauncher
from utils.automation.example.queue_creator import queue_path


if __name__ == '__main__':
    # queue_path = same queue_path as before
    queue = QueueManager(queue_path, make_directories=False)
    launcher = JobLauncher(queue)
    launcher.run()

    # for parallel processing launch multiple instances of this script