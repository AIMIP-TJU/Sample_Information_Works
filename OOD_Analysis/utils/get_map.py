import numpy as np
import pandas as pd

def get_map(file_name):
    file_name_list = pd.read_csv(file_name[0], header=None)
    label = file_name_list.values[:, 1]
    label = np.unique(label)
    label_map = {}
    for i, l in enumerate(label):
        label_map[l] = i
    return label_map