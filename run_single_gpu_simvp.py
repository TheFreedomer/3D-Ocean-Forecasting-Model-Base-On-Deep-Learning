import os.path
import pickle
import torch
from package.utils.MyDataset import Dataset
from package.utils.utils import training_single_gpu, load_checkpoint, delete_all_contents
from package.utils.Cost import MSE
from torch.utils.data import Subset, DataLoader
from datetime import datetime
from package.SimVP_Simpler_yet_Better_Video_Prediction_master.model import SimVP
import numpy as np


def execute(device,
            dataloader_train, dataloader_test,
            epochs, model, optimizer, lr, loss_fn, epoch_start,
            log_dir, record_step,
            filepath_checkpoint):
    # # Divide dataset
    t_s = datetime.now()
    training_single_gpu(device,
                        epochs,
                        model,
                        optimizer, lr,
                        loss_fn,
                        dataloader_train, dataloader_test,
                        epoch_start,
                        log_dir, record_step, filepath_checkpoint=filepath_checkpoint)
    t_e = datetime.now()
    print(f"耗时: {(t_e - t_s).total_seconds():.6f} 秒")


if __name__ == '__main__':
    # # start parameters
    device_ = torch.device(1)
    start_type = 'cold'     # 'cold' or 'warm'

    #
    lst_lead = [1]
    time_axis_ = np.load('./data/info_axis/time.npy')
    dir_dataset_ = r'./data/dataset'
    filepath_statistics_ = r'./data/statistics/all/statistics.pkl'

    # all
    # lst_variable_ = ['tx', 'ty', 'ssr', 'str',
    #                  'ssh',
    #                  'u', 'v', 't', 's']
    # lst_is_mask_ = [False, False, False, False,
    #                 True, True, True, True, True, ]
    # lst_level_ = [None, None, None, None, None,
    #               list(range(39)), list(range(39)), list(range(39)), list(range(39))]

    # part
    lst_variable_ = ['ssh']
    lst_is_mask_ = [True]
    lst_level_ = [None]

    dim_level_cat_ = 'variable'


    def transform_x(x):
        """

        :param x:
        :return:
        """
        x = np.ascontiguousarray(x[np.newaxis, :])

        return x


    def transform_y(y):
        """

        :param y:
        :return:
        """
        y = np.ascontiguousarray(y[np.newaxis, :])
        return y

    for lead_ in lst_lead:

        # # dataset parameters
        break_percent_ = 0.95
        batch_size_ = 80
        shuffle_ = True
        drop_last_ = True
        num_workers_ = 1

        # # dataset
        dataset = Dataset(lead=lead_, time_axis=time_axis_, dir_dataset=dir_dataset_,
                          filepath_statistics=filepath_statistics_, lst_variable=lst_variable_,
                          lst_is_mask=lst_is_mask_,
                          lst_level=lst_level_, dim_level_cat=dim_level_cat_,
                          transform_x=transform_x, transform_y=transform_y)

        # # divide dataset
        indices = list(range(len(dataset)))
        break_point = int(len(dataset) * break_percent_)
        indices_train, indices_test = indices[: break_point], indices[break_point:]
        dataset_train = Subset(dataset, indices_train)
        dataset_test = Subset(dataset, indices_test)
        dataloader_train_ = DataLoader(dataset_train, batch_size=batch_size_,
                                      shuffle=shuffle_, drop_last=drop_last_, num_workers=num_workers_)
        dataloader_test_ = DataLoader(dataset_test, batch_size=1,
                                     shuffle=False, drop_last=False, num_workers=0)

        # # model parameters
        img_size_ = 224
        in_chans_ = 1
        num_classes_ = 1
        seq_len = 1

        # # log parameters
        log_dir_ = os.path.join(r'result/logs', 'test' + str(lead_))
        record_step_ = 1  # one batch per step

        # # train parameters
        epochs_ = 10
        shape_in_ = (seq_len, in_chans_, img_size_, img_size_)
        channel_out_ = num_classes_
        hid_S_ = 16
        hid_T_ = 128
        N_S_ = 4
        N_T_ = 8
        groups_ = 8
        model_ = SimVP(shape_in=shape_in_, channel_out=channel_out_,
                       hid_S=hid_S_, hid_T=hid_T_, N_S=N_S_, N_T=N_T_, groups=groups_)
        model_.to(device_)
        lr_ = 0.1
        optimizer_ = torch.optim.Adam(model_.parameters(), lr=lr_)
        # loss_fn_ = torch.nn.MSELoss()
        loss_fn_ = MSE()

        #
        filepath_checkpoint_ = os.path.join(r'result/checkpoint',
                                            'test' + str(lead_) + '.pth')
        if start_type == 'warm':
            checkpoint = load_checkpoint(filepath_checkpoint_, model_, optimizer_, device_)
            epoch_start_ = checkpoint['epoch_start']
        elif start_type == 'cold':
            delete_all_contents(log_dir_)
            epoch_start_ = 0
        else:
            raise ValueError("start_type must be 'warm' or 'cold'!")

        # # save parameters
        # lead = lead_, time_axis = time_axis_, dir_dataset = dir_dataset_,
        # filepath_statistics = filepath_statistics_, lst_variable = lst_variable_,
        # lst_is_mask = lst_is_mask_,
        # lst_level = lst_level_, dim_level_cat = dim_level_cat_

        dict_config_model = {'model_init': model_.module.state_dict() if isinstance(model_, torch.nn.DataParallel)
        else model_.state_dict(),
                             'shape_in': shape_in_,
                             'channel_out': channel_out_,
                             'hid_S': hid_S_, 'hid_T': hid_T_,
                             'N_S': N_S_, 'N_T': N_T_,
                             'groups': groups_}

        dict_config_dataset = {'lead': lead_,
                               'time_axis': time_axis_,
                               'dir_dataset': dir_dataset_,
                               'filepath_statistics': filepath_statistics_,
                               'lst_variable': lst_variable_,
                               'lst_is_mask': lst_is_mask_,
                               'lst_level': lst_level_,
                               'dim_level_cat': dim_level_cat_,
                               'break_percent': break_percent_,
                               'batch_size': batch_size_,
                               'shuffle': shuffle_,
                               'drop_last': drop_last_,
                               'num_workers': num_workers_}

        dict_config_train = {'epochs': epochs_,
                             'lr': lr_,
                             'optimizer': optimizer_,
                             'loss_fn': loss_fn_}

        dict_config_log = {'log_dir': log_dir_,
                           'record_step': record_step_}

        dict_config = {'model': dict_config_model,
                       'dataset': dict_config_dataset,
                       'train': dict_config_train,
                       'log': dict_config_log}
        # filepath_config_ = os.path.join(r'./result/config', 'test_mg.pkl')
        filepath_config_ = os.path.join(r'result/config',
                                        'test' + str(lead_) + '.pkl')

        if start_type == 'cold':
            with open(filepath_config_, 'wb') as f:
                pickle.dump(dict_config, f)

        # # execute
        execute(device_,
                dataloader_train_, dataloader_test_,
                epochs_, model_, optimizer_, lr_, loss_fn_, epoch_start_,
                log_dir_, record_step_,
                filepath_checkpoint_)
