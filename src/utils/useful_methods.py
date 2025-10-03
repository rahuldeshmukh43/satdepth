import os
import pickle, json
import time
import logging
from functools import wraps

def setup_logger(name: str, log_file: str, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def write_args(args, f):
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    file.close
    return

def write_config(config, filename):
    s = json.dumps(config, indent=4)
    with open(filename, "w") as file:
        file.write(s)
    file.close()
    return

def get_basename(full_name):
    return os.path.basename(full_name).split('.')[0]

def get_img_path(filename):
    path = os.path.split(filename)[0]
    return path

def get_file_ext(filename):
    ext = os.path.split(filename)[1].split('.')[1]
    return ext

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pkl_write(filename, obj):
    f = open(filename, "wb")
    pickle.dump(obj, f)
    f.close()
    return

def pkl_read(filename):
    f = open(filename, "rb")
    obj = pickle.load(f)
    f.close()
    return obj
