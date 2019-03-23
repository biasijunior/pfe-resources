import tables

#h5_file = tables.open_file('my_arrays.h5', mode='w')
h5_file = tables.open_file('my_array.h5', mode='r', titel='many large arrays')

for node in h5_file:

    print(node)