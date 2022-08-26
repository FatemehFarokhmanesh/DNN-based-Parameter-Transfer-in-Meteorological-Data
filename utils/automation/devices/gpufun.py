import subprocess
import os

# adapted from Christian Reinbold


tolerable_processes = [
    '/usr/lib/xorg/Xorg'
]


def call_nvidia_smi(print_output=False):
    result_bstr = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    result_str = result_bstr.stdout.decode('utf-8')
    lines = [s[1:-1].strip() for s in result_str.split('\n')]
    if len(lines) < 4:
        raise Exception('[ERROR] Got unexpected program response (likely CUDA driver error)')
    if print_output:
        print('[INFO] NVIDIA-SMI Output:')
        print('\n'.join(lines))
    return lines


def find_cuda_devices():
    smi_str = call_nvidia_smi()
    sep_iter = iter(i for i, line in enumerate(smi_str) if line.startswith('='))
    start_gpu_block = next(sep_iter)
    start_process_block = next(sep_iter)
    all_devices = set(read_gpu_block(smi_str[start_gpu_block:start_process_block]))
    occupied_devices = set(read_process_block(smi_str[start_process_block:]))
    return all_devices, occupied_devices


def read_gpu_block(block):
    for line in block:
        if len(line) == 0:
            break
        entry = line.split()[0].strip()
        if entry.isnumeric():
            yield entry
        else:
            continue


def read_process_block(block):
    for line in block:
        if len(line) == 0:
            break
        entries = [e.strip() for e in line.split()]
        if entries[0].isnumeric() and entries[5] not in tolerable_processes:
            yield entries[0]
        else:
            continue


def set_free_devices_as_visible(num_devices=1):
    all_devices, occupied_devices = find_cuda_devices()
    free_devices = list(all_devices - occupied_devices)
    if len(all_devices) > 1:
        if len(free_devices) < num_devices:
            raise RuntimeError('[ERROR] Not enough cuda devices available.')
        visible_devices = free_devices[:num_devices]
        devices_str = ','.join(visible_devices)
    else:
        if len(free_devices) < num_devices:
            print('[WARN] Using only available CUDA device though it is occupied.')
        devices_str = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
    return devices_str


if __name__ == '__main__':
    print(find_free_cuda_devices())