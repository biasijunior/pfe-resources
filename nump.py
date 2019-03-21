import csv
import cv2

my_data = {'a': [1, 2.0, 3, 4+6j],
           'b': ('string', u'Unicode string'),
           'c': None}
with open('names.csv', 'w') as csvfile:
    fieldnames = ['first_name','last_name', 'something']
    # spamwriter = csv.writer(csvfile, delimiter=' ',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow(
        {'first_name': my_data, 'last_name': 'Beans', 'something': 'person'})
    writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
    writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
