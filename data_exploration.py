# Reading CSV File
target_file = open('dataset/pressure.csv', 'r')

line = target_file.readline()
headers = line[:-1].split(',')

dataset = {}

for header in headers :
    dataset[header] = []


while line != '' :
    line = target_file.readline()

    items = line[:-1].split(',')
    
    index = 0

    for key in dataset.keys() :
        dataset[key].append(items[index])
        index += 1 

        if index == len(items) :
            break

target_file.close()

print(dataset)