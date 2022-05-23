import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


data = '20210602'
id = '174612'

title = pd.read_table('/home/c611/projects/gxl/fewshot/Logs/' + data + '/' + id + '.log', nrows=8, header=0, sep=" ")

df = pd.read_table('/home/c611/projects/gxl/fewshot/Logs/' + data + '/' + id + '.log', header=9, sep=" ")

df_train_loss = df[df['Type'] == "train_loss"]
df_test_loss = df[df['Type'] == "test_loss"]
df_train_acc = df[df['Type'] == "train_acc"]
df_test_acc = df[df['Type'] == "test_acc"]

plt.subplot(221)
plt.plot(df_train_loss['num1'].values, alpha=0.3)
plt.plot(np_move_avg(df_train_loss['num1'].values, int(len(df_train_loss['num1'].values) / 50), mode='valid'))
plt.title('train_loss')
plt.subplot(222)
plt.plot(df_test_loss['num1'].values, alpha=0.3)
plt.plot(np_move_avg(df_test_loss['num1'].values, int(len(df_test_loss['num1'].values) / 50), mode='valid'))
plt.title('test_loss')
plt.subplot(223)
plt.plot(df_train_acc['num1'].values, alpha=0.3)
plt.plot(np_move_avg(df_train_acc['num1'].values, int(len(df_train_loss['num1'].values) / 50), mode='valid'))
plt.title('train_acc')
plt.subplot(224)
plt.plot(df_test_acc['num1'].values, alpha=0.3)
plt.plot(np_move_avg(df_test_acc['num1'].values, int(len(df_test_loss['num1'].values) / 50), mode='valid'))
plt.title('test_acc')


plt.suptitle(
    data + '/' + id + ' ' +
    title[title['info'] == 'dataset_name']['content'].values + ' ' +
    title[title['info'] == 'classifier_name']['content'].values + ' ' +
    title[title['info'] == 'method_name']['content'].values + ' ' +
    title[title['info'] == 'proportion']['content'].values
)

#plt.show()
plt.savefig('/home/c611/projects/gxl/fewshot/Logs/images/'+ id + '.jpg')
print(1)
