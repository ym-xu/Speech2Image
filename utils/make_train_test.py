import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from random import sample

x_test = []
x_train = []
cls_train = []
cls_test = []
all_cls = []
all_file = []
file_ads = './../data/birds/CUB_200_2011_audio/audio'

l = [i for i in range(1,201)] 
testset = sample(l, 50)

cls_names = os.listdir(file_ads)
i = 0
for cls_name in sorted(cls_names):
    cls_name = cls_name
    cls_path = os.path.join(file_ads, cls_name) 
    img_names = os.listdir(cls_path)
    for img_name in sorted(img_names):
      if cls_name + '/' + img_name[:-6] not in all_file:
         all_file.append(cls_name + '/' + img_name[:-6])
         if int(cls_name[:3]) not in testset:
            x_train.append(cls_name + '/' + img_name[:-6])
            cls_train.append(int(cls_name[:3]))
            # for i in range(10):
            #    x_train.append(cls_name + '/' + img_name[:-6] + '-' + str(i))
            #    cls_train.append(int(cls_name[:3]))
         else:
            x_test.append(cls_name + '/' + img_name[:-6])
            cls_test.append(int(cls_name[:3]))
            # for i in range(10):
            #    x_test.append(cls_name + '/' + img_name[:-6] + '-' + str(i))
            #    cls_test.append(int(cls_name[:3]))
            # print(cls_name + '/' + img_name[:-6] + '  ' + cls_name[:3])
    # if img_name[-2:] == '_4':
    #     #print(img_name[:-2])
    #     x_test.append(img_name)
    # else:
    #     #print(img_name[:-2])
    #     x_train.append(img_name)

# x_train, x_test, cls_train, cls_test = train_test_split(all_file, all_cls, test_size=0.2)


#train_df = pd.DataFrame(x_train)
#test_df = pd.DataFrame(x_test)
print(len(x_train), len(x_test))
print(len(cls_train), len(cls_test))

# # unique_id = np.unique(cls_train)
with open('./../data/birds/train-or/filenames.pickle', 'wb') as f:
   pickle.dump(x_train, f)

with open('./../data/birds/test-or/filenames.pickle', 'wb') as f:
   pickle.dump(x_test, f)

with open('./../data/birds/train-or/class_info.pickle', 'wb') as f:
   pickle.dump(cls_train, f)

with open('./../data/birds/test-or/class_info.pickle', 'wb') as f:
   pickle.dump(cls_test, f)

# for i in x_test:
#    if i[:3] == '001':
#       print(i)