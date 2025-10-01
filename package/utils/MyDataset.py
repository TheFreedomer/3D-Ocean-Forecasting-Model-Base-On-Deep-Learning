import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """

    """

    def __init__(self, filepath=r'../../data/dataset/data/ssh',
                 lead=1,
                 transform_x=None, transform_y=None,
                 filepath_mean=r'../../data/statistics/test/mean_data_merge.npy',
                 filepath_std=r'../../data/statistics/test/mean_data_merge.npy', ):
        """

        :param filepath:
        :param lead:
        :param transform_x:
        :param transform_y:
        :param filepath_mean:
        :param filepath_std:
        """
        self.filepath = filepath
        self.lead = lead
        self.transform_x = transform_x
        self.transform_y = transform_y
        #
        self.filenames = sorted(os.listdir(filepath))
        self.file_num = len(self.filenames)
        #
        self.mean_value = np.load(filepath_mean)
        self.std_value = np.load(filepath_std)

    def __len__(self):
        """

        :return:
        """
        return self.file_num - self.lead
        # return 100

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        #
        x = np.load(os.path.join(self.filepath, self.filenames[item]))
        y = np.load(os.path.join(self.filepath, self.filenames[item + self.lead]))
        #
        x[x == -32767] = np.nan
        y[y == -32767] = np.nan
        #
        x = np.transpose(x, (1, 2, 0))
        x = (x - self.mean_value) / self.std_value
        x = np.transpose(x, (2, 0, 1))
        #
        x[np.isnan(x)] = 0
        y[np.isnan(y)] = 0
        #
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class Dataset(Dataset):
    """

    """

    def __init__(self,
                 lead=1,
                 time_axis=None,
                 dir_dataset=r'../../data/dataset',
                 filepath_statistics=r'../../data/statistics/all/statistics.pkl',
                 lst_variable_feature=None,
                 lst_is_mask_feature=None,
                 lst_level_feature=None,
                 lst_variable_label=None,
                 lst_is_mask_label=None,
                 lst_level_label=None,
                 dim_level_cat='variable',
                 transform_x=None, transform_y=None):
        """

        :param lead:
        :param time_axis:
        :param dir_dataset:
        :param filepath_statistics:
        :param lst_variable_feature:
        :param lst_is_mask_feature:
        :param lst_level_feature:
        :param lst_variable_label:
        :param lst_is_mask_label:
        :param lst_level_label:
        :param dim_level_cat:
        :param transform_x:
        :param transform_y:
        """
        self.lead = lead
        self.time_axis = time_axis
        self.dir_dataset = dir_dataset
        self.filepath_statistics = filepath_statistics
        self.lst_variable_feature = lst_variable_feature
        self.lst_is_mask_feature = lst_is_mask_feature
        self.lst_level_feature = lst_level_feature

        self.lst_variable_label = lst_variable_label
        self.lst_is_mask_label = lst_is_mask_label
        self.lst_level_label = lst_level_label

        self.dim_level_cat = dim_level_cat
        self.transform_x = transform_x
        self.transform_y = transform_y
        #
        assert self.dim_level_cat in ['sequence', 'variable']
        if dim_level_cat == 'sequence':
            assert all(lst_level_feature) is True
        #
        with open(os.path.join(self.filepath_statistics), 'rb') as f:
            self.dict_statistics = pickle.load(f)
        self.filename_prefix = np.datetime_as_string(time_axis, unit='D')
        pass

    def __len__(self):
        """


        :return:
        """

        return self.filename_prefix.size - self.lead

    def get_data(self, item, mean_value, std_value, dir_data, dir_mask=None, norm=False):
        """

        :param item:
        :param mean_value:
        :param std_value:
        :param dir_data:
        :param dir_mask:
        :param norm:
        :return:
        """
        #
        data = np.load(os.path.join(dir_data, self.filename_prefix[item] + '.npy'))
        #
        if norm:
            data = (data - mean_value) / std_value

        if dir_mask:
            #
            mask = np.load(os.path.join(dir_mask, self.filename_prefix[item] + '.npy'))
            #
            data[mask] = 0

        return data

    def merge_data(self, item, lst_variable, lst_is_mask, lst_level, norm):
        """"""
        #
        lst_merge = []
        #
        for v, variable in enumerate(lst_variable):
            if lst_is_mask[v]:
                if lst_level[v]:
                    for level in lst_level[v]:
                        dir_name = 'level' + '{:02d}'.format(level)
                        dir_final_data = os.path.join(self.dir_dataset, 'data',
                                                      variable, dir_name)
                        dir_final_mask = os.path.join(self.dir_dataset, 'mask',
                                                      variable, dir_name)
                        mean_value = self.dict_statistics[variable][dir_name]['mean']
                        std_value = self.dict_statistics[variable][dir_name]['std']
                        data = self.get_data(item, mean_value, std_value,
                                             dir_final_data, dir_final_mask, norm)
                        lst_merge .append(data)
                else:
                    dir_final_data = os.path.join(self.dir_dataset, 'data', variable)
                    dir_final_mask = os.path.join(self.dir_dataset, 'mask', variable)
                    mean_value = self.dict_statistics[variable]['mean']
                    std_value = self.dict_statistics[variable]['std']
                    data = self.get_data(item, mean_value, std_value,
                                         dir_final_data, dir_final_mask, norm)
                    lst_merge.append(data)
            else:
                dir_final_data = os.path.join(self.dir_dataset, 'data', variable)
                dir_final_mask = None
                mean_value = self.dict_statistics[variable]['mean']
                std_value = self.dict_statistics[variable]['std']
                data = self.get_data(item, mean_value, std_value,
                                     dir_final_data, dir_final_mask, norm)
                lst_merge.append(data)
        #
        data_merge = np.concatenate(lst_merge, axis=0)

        return data_merge

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        x = self.merge_data(item, self.lst_variable_feature, self.lst_is_mask_feature,
                            self.lst_level_feature, norm=True)
        y = self.merge_data(item + self.lead, self.lst_variable_label, self.lst_is_mask_label,
                            self.lst_level_label, norm=False)
        #
        if self.dim_level_cat == 'sequence':
            _, img_size, _ = x.shape
            seq_len = len(self.lst_level_feature[0])
            variable_len = len(self.lst_variable_feature)
            x = x.reshape((variable_len, seq_len, img_size, img_size)).transpose((1, 0, 2, 3))
            y = y.reshape((variable_len, seq_len, img_size, img_size)).transpose((1, 0, 2, 3))
        #
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)

        return x.astype(np.float32), y.astype(np.float32)


if __name__ == '__main__':
    #
    lead_ = 1
    time_axis_ = np.load('../../data/info_axis/time.npy')
    dir_dataset_ = r'../../data/dataset'
    filepath_statistics_ = r'../../data/statistics/all/statistics.pkl'

    # all
    # lst_variable_ = ['tx', 'ty', 'ssr', 'str',
    #                  'ssh',
    #                  'u', 'v', 't', 's']
    # lst_is_mask_ = [False, False, False, False,
    #                 True, True, True, True, True, ]
    # lst_level_ = [None, None, None, None, None,
    #               list(range(39)), list(range(39)), list(range(39)), list(range(39))]

    # part
    depth_level = 39
    lst_variable_feature_ = ['u', 'v', 't', 's']
    lst_is_mask_feature_ = [True, True, True, True, ]
    lst_level_feature_ = [list(range(depth_level)), list(range(depth_level)),
                          list(range(depth_level)), list(range(depth_level))]
    idx_label_s = 0
    lst_variable_label_ = lst_variable_feature_[idx_label_s:]
    lst_is_mask_label_ = lst_is_mask_feature_[idx_label_s:]
    lst_level_label_ = lst_level_feature_[idx_label_s:]

    dim_level_cat_ = 'sequence'
    transform_x_ = None
    transform_y_ = None
    dataset = Dataset(lead=lead_, time_axis=time_axis_, dir_dataset=dir_dataset_,
                      filepath_statistics=filepath_statistics_,
                      lst_variable_feature=lst_variable_feature_,
                      lst_is_mask_feature=lst_is_mask_feature_,
                      lst_level_feature=lst_level_feature_,
                      lst_variable_label=lst_variable_label_,
                      lst_is_mask_label=lst_is_mask_label_,
                      lst_level_label=lst_level_label_,
                      dim_level_cat=dim_level_cat_)
    print(dataset[0][1].shape)
