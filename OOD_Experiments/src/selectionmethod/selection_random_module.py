
class select_random(object):
    def __init__(self, proportion):
        self.proportion = proportion

    def select(self, file_name_list):

        file_name_list_selected = file_name_list.sample(frac=self.proportion).reset_index(drop=True)

        return file_name_list_selected


"""
import pandas as pd


select_method = select_random(0.7)
file_name_list = pd.read_csv('D:/Projects/FewShot/datasets/correlate.csv', header=None)
file_name_list_selected = select_method.select(file_name_list)
index = 5
name = file_name_list_selected.values[index, 0]
label = file_name_list_selected.values[index, 1]
print(2)"""
