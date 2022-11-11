import random
import csv

N = 1500 #training
M = 150 #validation
list = random.sample(range(0, 15050), N+M)

training_list = list[:N]
validation_list = list[N:]

tr_name = []
vl_name = []
with open('/home/lorenzo/Documents/unimg/ML/Data/training_set_Rosetta/training_set/list15051.txt', 'r') as text:
    count = 0
    for line in text.readlines():
        line = line.replace('\n', '')
        if count in training_list:
            tr_name.append([count, line])
        elif count in validation_list:
            vl_name.append([count, line])
        count += 1

with open('/home/lorenzo/Documents/unimg/ML/training_set.csv', 'w') as ts:
    ts_writer = csv.writer(ts, delimiter=',', quotechar='"')
    ts_writer.writerow(['ID', 'NAME'])
    for line in range(len(tr_name)):
        ts_writer.writerow(tr_name[line])

with open('/home/lorenzo/Documents/unimg/ML/validation_set.csv', 'w') as vs:
    vs_writer = csv.writer(vs, delimiter=',', quotechar='"')
    vs_writer.writerow(['ID', 'NAME'])
    for line in range(len(vl_name)):
        vs_writer.writerow(vl_name[line])




