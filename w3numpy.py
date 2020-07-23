# I am not sure I understand array transposition, but this is the gist of what I got

import numpy as np

arr = np.arange(12)
reshaped = np.reshape(arr, (2,3,2))
print("This is the original array: ")
print(arr)
print("Thisis the reshaped array: ")
print(reshaped)

flattened = reshaped.flatten()
transposed = np.transpose(reshaped, axes=(1,2,0))
print("This is the flattened array: ")
print(flattened)
print("This is the transposed array, (2,3,2) transposed by (1,2,0) should give 3 groups of 2 rows and 2 columns: ")
print(transposed)

transpose_df = np.transpose(reshaped, axes=(2,1,0))
print("This same array (2,3,2) transposed by (2,1,0) should give us 2 groups of 3 rows and 2 columns")
print(transpose_df)