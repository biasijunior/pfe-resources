import numpy as np
import tables

my_arrays = [np.ones((2, 2, 4098)) for x in range(10)]

h5_file = tables.open_file('my_array.h5', mode='w', titel='many large arrays')

for n, arr in enumerate(my_arrays):
    h5_file.create_array('/', 'my_array{}'.format(n), arr)
h5_file.close()
