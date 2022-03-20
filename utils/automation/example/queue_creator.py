from utils.automation.queue import QueueManager, JobMaker

# indicate script path
script_path = './script.py'

# indicate interpreter path
interpreter_path = '/home/frokhma/anaconda3/envs/py38/bin/python' # adapt that for your purposes

# indicate where to setup the queue
queue_path = './queue'


if __name__ == '__main__':
    # set up queue and get QueueManager
    queue = QueueManager(queue_path, make_directories=True)
    # be careful with directory making,especially when using relative path definitions
    # --> folders might not end up being made where they are expected to be
    # get job makers that return a single run configuration
    job_maker_cpu = JobMaker(interpreter_path, script_path)
    # don't add flags --> use default CPU computation
    job_maker_gpu = JobMaker(interpreter_path, script_path, flags=['--gpu-enabled']) # note the brackets!
    # add flag '--gpu-enabled' --> use GPU

    # get job maker that returns 2 different job configurations
    choices = ['--gpu-enabled', '--gpu-disabled']
    job_maker_both = JobMaker(interpreter_path, script_path, flags=[choices]) # note the brackets also here!
    # Remark concerning the brackets:
    # For multiple flags and multiple choices the code would be
    # choices_flag1 = ['--flag1-enabled', '--flag1-disabled]
    # choices_flag2 = ['--flag2-enabled', '--flag2-disabled]
    # job_maker_multi = JobMaker(interpreter_path, script_path, flags=[choices_flag1, choices_flag2])
    # --> job maker returns 4 configurations, resulting from combinations of the above lists:
    # --> [(enabled,enabled), (enabled,disabled), (disabled, enabled), (disabled, disabled)]
    # handling of positional or keyword arguments is similar

    # push job to the queue, repeat a few times to get multiple jobs waiting
    job_maker_cpu.export_jobs(queue)
    job_maker_gpu.export_jobs(queue)
    job_maker_both.export_jobs(queue)
    # queue should now contain 4 jobs
    # Remark: The queue is a database that stores the jobs until they are processed or deleted
    # --> to clean up the queue either deleat the queue directory completely or use queue.remove_jobs(...)
    print('[INFO] Pushed jobs to queue')
