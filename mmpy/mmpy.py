import numpy as np
from os.path import isfile
from os import remove
from time import sleep
import json


def delete(filename):
    """
        removes the file from disk
    :param filename:
    :return:
    """
    assert isfile(filename)
    name = filename[0:-4] if filename.endswith('.npy') else filename
    lookup_file = name + '_lookup.npy'
    assert isfile(lookup_file)
    meta_file = name + 'meta.json'
    assert isfile(meta_file)
    remove(filename)
    remove(meta_file)
    remove(lookup_file)
    sleep(0.01)


class Reader:

    def __init__(self, filename):
        """
            reads the data
        """
        assert isfile(filename)
        name = filename[0:-4] if filename.endswith('.npy') else filename
        lookup_file = name + '_lookup.npy'
        assert isfile(lookup_file)
        meta_file = name + 'meta.json'
        assert isfile(meta_file)
        lookup = np.load(lookup_file)
        with open(meta_file, 'r') as f:
            meta = json.loads(''.join(f.readlines()))

        n = meta['n']
        max_shape = meta['max_shape']
        dtype = meta['dtype']

        shape = (n, *max_shape)
        X = np.memmap(filename, shape=shape, dtype=dtype, mode='r')
        self.X = X
        self.lookup = lookup

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        if isinstance(item, (list, slice, np.ndarray)):
            if isinstance(item, slice):
                step = 1 if item.step is None else item.step
                item = range(item.start, item.stop, step)

            result = []
            for i in item:
                h, w, c = self.lookup[i]
                im = self.X[i, 0:h, 0:w, 0:c]
                result.append(im)
            return result
        elif isinstance(item, int):
            h, w, c = self.lookup[item]
            return self.X[item, 0:h, 0:w, 0:c]
        else:
            raise ValueError("Cannot get ", item)


class Writer:

    def __init__(self, filename, n, max_shape, dtype, overwrite=False):
        """
            writes the data into a memory-mapped file
        :param filename: {string}
        :param n: {int} #files
        :param max_shape: (h,w,c)
        :param dtype: numpy dtype
        :param overwrite: {boolean} if True: delete old data first
        """
        assert n > 0

        name = filename[0:-4] if filename.endswith('.npy') else filename
        lookup_file = name + '_lookup.npy'
        meta_file = name + 'meta.json'

        shape = (n, *max_shape)
        if isfile(filename) and not overwrite:
            assert isfile(lookup_file)
            assert isfile(meta_file)
            with open(meta_file, 'r') as f:
                meta = json.loads(''.join(f.readlines()))
            assert len(meta['max_shape']) == len(max_shape)
            for a, b in zip(meta['max_shape'], max_shape):
                assert a == b
            assert meta['n'] == n
            assert meta['dtype'] == dtype

            lookup = np.load(lookup_file)
            assert len(lookup) == n
            left = lookup[:, 0]
            M = ((left < 0) * 1).nonzero()[0]
            if len(M) == 0:
                current_pointer = -1  # list is full
            else:
                current_pointer = M[0]
            del M

            X = np.memmap(filename, shape=shape, dtype=dtype, mode='r+')
        else:
            meta = {
                "max_shape": max_shape,
                "n": n,
                "dtype": dtype
            }
            with open(meta_file, 'w+') as f:
                meta_dmp = json.dumps(meta)
                f.write(meta_dmp)

            current_pointer = 0
            lookup = np.ones((n, 3), 'int32') * -1
            X = np.memmap(filename, shape=shape, dtype=dtype, mode='w+')

        self.current_pointer = current_pointer
        self.lookup = lookup
        self.meta_file = meta_file
        self.lookup_file = lookup_file
        self.X = X
        self.max_shape = max_shape
        self.n = n

    def add(self, datum):
        """
            adds a datum to the end of the function
        :param datum:
        :return:
        """
        if self.current_pointer < 0:
            raise BufferError("out of bounds")
        assert self.current_pointer < self.n
        self.insert(self.current_pointer, datum)
        curp = self.current_pointer + 1
        self.current_pointer = curp if curp < self.n else -1

    def insert(self, i, datum):
        """
            inserts a new datum into the set
        :param i:
        :param datum:
        :return:
        """
        assert i < self.n
        assert self.lookup[i, 0] == -1 and self.lookup[i, 1] == -1
        assert len(datum.shape) == len(self.max_shape)
        for a, b in zip(datum.shape, self.max_shape):
            assert a <= b
        h, w, c = datum.shape
        self.X[i, 0:h, 0:w, 0:c] = datum
        self.lookup[i, 0] = h
        self.lookup[i, 1] = w
        self.lookup[i, 2] = c

    def flush(self):
        if self.X is not None:
            del self.X  # flush data
        try:
            remove(self.lookup_file)
            sleep(0.01)
        except:
            pass
        np.save(self.lookup_file, self.lookup)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
