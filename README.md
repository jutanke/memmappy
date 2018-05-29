# memmappy
Simple helper function to store large amounts of image-like things in numpy memory-mapped files for easy and fast'ish access.
This library is absolutely not disk space efficient and has very little features as it is geared towards my specific use-cases, use at your own risk.

## Install
```bash
pip install git+https://github.com/jutanke/memmappy.git
```

## Create memmapped file
```python
import numpy as np
from mmpy import mmpy as mm

fname = '/tmp/test.npy'
n = 10
max_shape = (10, 12, 3)
dtype = 'uint8'

# --- some dummy data ---
A = np.ones((10, 12, 3))
B = np.ones((10, 10, 3)) * 2
C = np.ones((9, 12, 3)) * 3
D = np.ones((5, 5, 1)) * 4

with mm.Writer(fname, n, max_shape, dtype) as f:
    for d in [A,B,C,D]:
        f.add(d)
        # f.insert(index, d)  # can be used to place an object at a specific location
```

## Load memmapped file
```python
from mmpy import mmpy as mm

fname = '/tmp/test.npy'
R = mm.Reader(fname)
A = R[0]  # access single element

List = R[[0, 2]]  # access a list

Slice = R[1:4]  # access a slice
```

## Delete memmapped file
```python
from mmpy import mmpy as mm

fname = '/tmp/test.npy'
mm.delete(fname)
```
