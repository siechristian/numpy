# Machine Learning course on Educative, powered by Adapt

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


arr = np.linspace(5, 11, num=4)
print(repr(arr))

arr = np.linspace(5, 11, num=4, endpoint=False) #float32 by default
print(repr(arr))

arr = np.linspace(5, 11, num=4, dtype=np.int32)
print(repr(arr))

# b. reshape data

arr = np.arange(8)

reshaped_arr = np.reshape(arr, (2, 4))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))

reshaped_arr = np.reshape(arr, (-1, 2, 2))  #wtf does the -1 do?
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))

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

