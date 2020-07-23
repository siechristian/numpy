# Machine Learning course on Educative, powered by Adapt

# 1.  Intro

import numpy as np  # import the NumPy library

# Initializing a NumPy array
arr = np.array([-1, 2, 5], dtype=np.float32)

# Print the representation of the array
print(repr(arr))

# 2. NumPy Arrays

# 2. a. Arrays

import numpy as np

arr = np.array([[0, 1, 2], [3, 4, 5]],
               dtype=np.float32)
print(repr(arr))


arr = np.array([0, 0.1, 2])
print(repr(arr))

# Output
0.668s
array([[0., 1., 2.],
       [3., 4., 5.]], dtype=float32)

#Output
0.904s
array([0., 0.1, 2.])


# 2.b Copying arrays

a = np.array([0, 1])
b = np.array([9, 8])
c = a
print('Array a: {}'.format(repr(a)))
c[0] = 5
print('Array a: {}'.format(repr(a)))

d = b.copy()
d[0] = 6
print('Array b: {}'.format(repr(b)))
print('Array b: {}'.format(repr(d)))

print("Output
0.483s
Array a: array([0, 1])
Array a: array([5, 1])
Array b: array([9, 8])
Arrayb: array([6, 8])")

# 2.c. Casting

arr = np.array([0, 1, 2])
print(arr.dtype)
arr = arr.astype(np.float32)
print(arr.dtype)

Output
1.216s
int64
float32

# 2.d.NaN 

arr = np.array([np.nan, 1, 2])
print(repr(arr))

arr = np.array([np.nan, 'abc'])
print(repr(arr))

# Will result in a ValueError
np.array([np.nan, 1, 2], dtype=np.int32)

# Will this result in a ValueError as well? Comment out previous line before running this. n.b. IT DID NOT!!!!!!!
np.array([np.nan, 1, 2], dtype=np.float32)

Output
0.503s
array([nan,  1.,  2.])
array(['nan', 'abc'], dtype='<U32')


Traceback (most recent call last):
  File "main.py", line 10, in <module>
    np.array([np.nan, 1, 2], dtype=np.int32)
ValueError: cannotconvertfloatNaNtointeger


# 2.e. Infinity

print(np.inf > 1000000)

arr = np.array([np.inf, 5])
print(repr(arr))

arr = np.array([-np.inf, 1])
print(repr(arr))

# Will result in an OverflowError. INFINITY IS FLOAT
np.array([np.inf, 3], dtype=np.int32)


Output
0.436s
True
array([inf,  5.])
array([-inf, 1.])
# commentedouttheinfinityportion

Traceback (most recent call last):
  File "main.py", line 12, in <module>
    np.array([np.inf, 3], dtype=np.int32)
OverflowError: cannot convert float infinity to integer


# 2.Test

# CODE HERE
arr = np.array([np.nan, 2, 3, 4,5])


# CODE HERE
arr = np.array([np.nan, 2, 3, 4,5])
arr2 = arr.copy()
arr2[0] = 10
print(arr2)

# CODE HERE
float_arr = np.array([1, 5.4, 3])
float_arr2 = arr2.astype(np.float32)

# CODE HERE
matrix = np.array([[1,2,3], [4, 5, 6]], dtype=np.float32)




# Numpy Basics

# a. ranged data

arr = np.arange(5)
print(repr(arr))

arr = np.arange(5.1) # decimal points from zero to 5
print(repr(arr))

arr = np.arange(3.3) #want to know what's up. decimal points up to 3. does not work with negative numbers
print(repr(arr))

arr = np.arange(-1, 4)
print(repr(arr))

arr = np.arange(-1.5, 4, 2)
print(repr(arr))

Output
0.655s
array([0, 1, 2, 3, 4])
array([0., 1., 2., 3., 4., 5.])
array([0., 1., 2., 3.])
array([-1,  0,  1,  2,  3])
array([-1.5, 0.5, 2.5])



arr = np.linspace(5, 11, num=4)
print(repr(arr))

arr = np.linspace(5, 11, num=4, endpoint=False) #float32 by default
print(repr(arr))

arr = np.linspace(5, 11, num=4, dtype=np.int32)
print(repr(arr))


Output
0.615s
array([ 5.,  7.,  9., 11.])
array([5. , 6.5, 8. , 9.5])
array([ 5,  7,  9, 11], dtype=int32)

# b. reshape data

arr = np.arange(8)

reshaped_arr = np.reshape(arr, (2, 4))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))

reshaped_arr = np.reshape(arr, (-1, 2, 2))  #wtf does the -1 do?
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))

0.425s
array([[0, 1, 2, 3], [4, 5, 6, 7]])Newshape: (2, 4)array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])Newshape: (2, 2, 2)

Output
0.400s
array([[0, 1, 2, 3],
       [4, 5, 6, 7]])
arr shape: (2, 4)
array([0, 1, 2, 3, 4, 5, 6, 7])
flattenedshape: (8,)



arr = np.arange(8)
arr = np.reshape(arr, (2, 4))
flattened = arr.flatten()
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(flattened))
print('flattened shape: {}'.format(flattened.shape))

# c. transposing data

arr = np.arange(8)
arr = np.reshape(arr, (4, 2))
transposed = np.transpose(arr)
print("original array: ")
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print("Transposed Array: ")
print(repr(transposed))
print('transposed shape: {}'.format(transposed.shape))


Output
0.429s
original array: 
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7]])
arr shape: (4, 2)
Transposed Array: 
array([[0, 2, 4, 6],
       [1, 3, 5, 7]])
transposedshape: (2, 4)




arr = np.arange(24)
arr = np.reshape(arr, (3, 4, 2))
transposed = np.transpose(arr, axes=(1, 2, 0))
print("This is the reshaped Array into 3 groups of 4 rows and 2 columns")
print(arr)
print('arr shape: {}'.format(arr.shape))
print("This is the transposed array: into 4 groups of 2 rows and 4 columns")
print(transposed)
print('transposed shape: {}'.format(transposed.shape))

transpose_df = np.transpose(arr, axes=(2, 1, 0))
print("The default transposed array is 2 groups of 4 rows and 3 columns")
print(transpose_df)
print('transpose_df shape: {}'.format(transpose_df.shape))

# d. zeros and ones

arr = np.zeros(4)
print(repr(arr))

arr = np.ones((2, 3))
print(repr(arr))

arr = np.ones((2, 3), dtype=np.int32)
print(repr(arr))

arr = np.array([[1, 2], [3, 4]])
print(repr(np.zeros_like(arr)))

arr = np.array([[0., 1.], [1.2, 4.]])
print(repr(np.ones_like(arr)))
print(repr(np.ones_like(arr, dtype=np.int32)))

# practice problems 
# CODE HERE
arr = np.arange(12)
reshaped = np.reshape(arr, (2, 3, 2))


# CODE HERE
flattened = reshaped.flatten()
transposed = np.transpose(reshaped, axes=(1,2,0))

# original array (2,3,2) transposed by (1,2,0) should give us 3-index1 groups of 2-index2 rows and 2-index0 columns
print(flattened)
print(transposed)

