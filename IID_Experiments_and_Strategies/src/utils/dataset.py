import numpy
import torch.utils.data as Data
import pandas as pd
import numpy as np
from PIL import Image
import csv
import random

def get_map(file_name_list):
    # print(file_name_list)
    label_list = np.unique(numpy.array(file_name_list)[:, 1])
    label_map = {}
    for i, l in enumerate(label_list):
        label_map[l] = i
    return label_map
class TrainDataset(Data.Dataset):
    """
    数据集
    """
    def __init__(self,select_ratio,base_csv_path, file_name_list, pool_list, file_path, file_name,num_classes, select_data, transform=None):
        """

        :param file_path:数据集位置
        :param file_name:样本索引目录
        :param target_map:数据集种类标签和实际标签间的映射
        :param transform:图像变换与增强

        """
        self.file_path = file_path  # '/root/disk/Zz/Dataset_rcs_structed/'
        # if select_method:
        #     self.file_name_list = select_method.select(pd.read_csv(file_name, header=None))
        # else:

        if select_data is None:
            with open(file_name, 'r') as f:
                files_reader = csv.reader(f)
                files = list(files_reader)
                number = np.zeros(num_classes)
                k=0
                for r in range(len(files)):
                    number[k] += 1
                    if r+1<len(files):
                        if files[r][1] != files[r+1][1] :
                            k += 1
                            assert k < len(number)
                number = number*select_ratio
                # random.shuffle(files)
                flag = np.zeros(num_classes)
                self.file_name_list = []
                self.pool_list = []
                label_list = np.unique(numpy.array(files)[:, 1])
                for file1 in files:
                    for i in range(num_classes):
                        if file1[1] == label_list[i]:
                            if flag[i] < number[i]:
                                self.file_name_list.append(file1)
                                flag[i] += 1
                            else:
                                self.pool_list.append(file1)

            ft = open(base_csv_path, 'w', newline='')
            ft_csv = csv.writer(ft)
            ft_csv.writerows(self.file_name_list)
            self.pool_list = np.array(self.pool_list)
            self.file_name_list = np.array(self.file_name_list)

        else:
            # print(select_data)
            self.file_name_list = np.concatenate((np.array(file_name_list), np.array(select_data)),axis=0)
            new_pool = []
            # print(len(select_data))
            # print(len(pool_list))
            o = 0
            for i in pool_list:
                flag = True
                if o != len(select_data):
                    for j in select_data:
                        if i[0] == j[0] and i[1] == j[1]:
                            flag = False
                            o += 1
                            break
                if flag:
                    new_pool.append(i)
            self.pool_list = np.array(new_pool)
            # print(len(self.pool_list))
        self.transform = transform
        label_map = get_map(self.file_name_list)
        self.target_map = label_map

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        image_path = self.file_path + self.file_name_list[index, 0]
        img_name = self.file_name_list[index, 0]
        img_lable = self.file_name_list[index, 1]
        # img = Image.open('D:/Projects/Datasets/example/di.jp')g
        # from PIL import ImageFile
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        # img = np.array(img).astype(np.float32)
        target = self.target_map[self.file_name_list[index, 1]]

        return img, target, img_name, img_lable

class TestDataset(Data.Dataset):
    """
    数据集
    """

    def __init__(self, file_path, file_name, transform=None):
        """

        :param file_path:数据集位置
        :param file_name:样本索引目录
        :param target_map:数据集种类标签和实际标签间的映射
        :param transform:图像变换与增强

        """
        self.file_path = file_path  # '/root/disk/Zz/Dataset_rcs_structed/'
        # if select_method:
        #     self.file_name_list = select_method.select(pd.read_csv(file_name, header=None))
        # else:
        self.file_name_list = pd.read_csv(file_name, header=None)
        self.transform = transform
        label_map = get_map(self.file_name_list.values)
        self.target_map = label_map

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        image_path = self.file_path + self.file_name_list.values[index, 0]
        img_name = self.file_name_list.values[index, 0]
        img_lable = self.file_name_list.values[index, 1]
        # img = Image.open('D:/Projects/Datasets/example/di.jp')g
        # from PIL import ImageFile
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        # img = np.array(img).astype(np.float32)
        target = self.target_map[self.file_name_list.values[index, 1]]

        return img, target, img_name, img_lable
class Pool(Data.Dataset):
    """
    数据集
    """

    def __init__(self,file_path, pool_list, transform=None):
        self.file_path = file_path
        self.file_name_list = pool_list
        self.transform = transform
        label_map = get_map(self.file_name_list)
        self.target_map = label_map

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        image_path = self.file_path + self.file_name_list[index, 0]
        img_name = self.file_name_list[index, 0]
        img_lable = self.file_name_list[index, 1]
        # img = Image.open('D:/Projects/Datasets/example/di.jp')g
        # from PIL import ImageFile
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        # img = np.array(img).astype(np.float32)
        target = self.target_map[self.file_name_list[index, 1]]

        return img, target, img_name, img_lable