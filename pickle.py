import cPickle as pickle

my_data = {'a': [1, 2.0, 3, 4+6j],
           'b': ('string', u'Unicode string'),
           'c': None}
output = open('data.pkl', 'wb')
pickle.dump(my_data, output)
pickle.dump(my_data, output)
output.close()
