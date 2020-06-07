import os
import datetime

def coalesce_none(*args):
    if len(args) == 0:
        raise ValueError('coalesce_none expects one or more positional arguments')
    
    for value in args:
        if value is not None:
            return value
    
    return args[-1]

def create_dir_if_missing(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def current_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')