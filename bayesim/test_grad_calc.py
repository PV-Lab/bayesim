import numpy as np

mat = np.random.randint(low=0,high=100,size=(5,5))
mat

mat.shape

deltas = {}
for i in range(len(mat.shape)):
    
