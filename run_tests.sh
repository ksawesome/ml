#!/bin/bash

# Minimal NumPy test: matrix multiplication
python3 -c "import numpy as np; a = np.random.rand(3,3); b = np.random.rand(3,3); result = np.dot(a, b); print('Matrix multiplication result:'); print(result)"