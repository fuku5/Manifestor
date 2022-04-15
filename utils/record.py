import gzip
import pickle
import atexit
import numpy as np
from multiprocessing import Manager

manager = Manager()
memory = None
out_name = 'record'

# record
def add(content):
    #print('add')
    memory.append(content)

def dump():
    print('dump')
    if memory is not None:
        with gzip.open('data/records/{}.pickle.gzip'.format(out_name), mode='wb') as f:
            f.write(pickle.dumps(list(memory)))

def init():
    print('init')
    global memory
    if memory is None:
        atexit.register(dump)
        print('recorded')
        memory = manager.list()


# use
def load(path='data/records/0211_record.pickle.gzip'):
    #print('load {}'.format(path))
    with gzip.open(path, mode='rb') as f:
        data = pickle.load(f)
    #print('done')
    return data

if __name__ == '__main__':
    import sys
    if len(sys.argv[1]) != 1:
        print(sys.argv[1])
        data = load(sys.argv[1])
    else:
        data = load()

    print(len(data))

