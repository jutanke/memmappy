import numpy as np
from os.path import isfile
from os import remove
from time import sleep


def delete(filename):
    """
        removes the file from disk
    :param filename:
    :return:
    """
    assert isfile(filename)
    name = filename[0:-4] if filename.endswith('.npy') else filename
    meta_file = name + '_meta.npy'
    assert isfile(meta_file)
    remove(filename)
    remove(meta_file)
    sleep(0.01)


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
        meta_file = name + '_meta.npy'

        shape = (n, *max_shape)
        if isfile(filename) and not overwrite:
            assert isfile(meta_file)
            meta = np.load(meta_file)
            assert len(meta) == n
            left = meta[:, 0]
            M = ((left < 0) * 1).nonzero()[0]
            if len(M) == 0:
                current_pointer = -1  # list is full
            else:
                current_pointer = M[0]
            del M

            X = np.memmap(filename, shape=shape, dtype=dtype, mode='r+')
        else:
            current_pointer = 0
            meta = np.ones((n, 2), 'int32') * -1
            X = np.memmap(filename, shape=shape, dtype=dtype, mode='w+')

        self.current_pointer = current_pointer
        self.meta = meta
        self.meta_file = meta_file
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
        assert self.meta[i, 0] == -1 and self.meta[i, 1] == -1
        assert len(datum.shape) == len(self.max_shape)
        for a, b in zip(datum.shape, self.max_shape):
            assert a <= b
        h, w, c = datum.shape
        self.X[i, 0:h, 0:w, 0:c] = datum

    def flush(self):
        if self.X is not None:
            del self.X  # flush data
        try:
            remove(self.meta_file)
            sleep(0.01)
        except:
            pass
        np.save(self.meta_file, self.meta)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
