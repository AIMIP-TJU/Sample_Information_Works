import pandas as pd
import numpy as np
import utils

trainset = pd.read_csv('../../datasets/mini-imagenet/train_t.csv', header=None, names=["name", "label"])
testset = pd.read_csv('../../datasets/mini-imagenet/test_t.csv', header=None, names=["name", "label"])
valset = pd.read_csv('../../datasets/mini-imagenet/val_t.csv', header=None, names=["name", "label"])
dataset = pd.concat([trainset, testset, valset])

label = dataset.values[:, 1]
label = np.unique(label)
df_list = []

for label_name in label:
    df_list.append(dataset[dataset["label"] == label_name])

test_list = []
train_list = []
for df in df_list[0:10]:
    df1 = df.sample(frac=0.2)
    test_list.append(df1)
    df2 = df[~df.index.isin(df1.index)]
    train_list.append(df2)

train_set = pd.concat(train_list)
test_set = pd.concat(test_list)
train_set.to_csv("train_new.csv", header=False, index=False )
test_set.to_csv("test_new.csv", header=False, index=False)

print(1)

