import csv

with open('images.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=':', quoting=csv.QUOTE_NONE)
    i = 0
    for row in reader:
        print row[0]
        desc2 = row[1]

        # print row[1]
        name = row[0]
        if name == 'Biasi':
            print row[0]
            print('equal')
