import os.path
import pickle
import torch
from package.utils.MyDataset import MyDataset
from package.utils.utils import training_multiple_gpu, load_checkpoint, delete_all_contents
from package.utils.Cost import MSE
from torch.utils.data import Subset, DataLoader
from datetime import datetime
from package.SwinUnet_main.networks.swin_transformer_unet_skip_expand_decoder_sys import \
    SwinTransformerSys


def execute(gpu_num,
            filepath, lead,
            break_percent, batch_size, shuffle, drop_last, num_workers,
            epochs, model, optimizer, lr, loss_fn, epoch_start,
            log_dir, record_step,
            filepath_checkpoint):
    """
    
    :param gpu_num: 
    :param filepath: 
    :param lead: 
    :param break_percent: 
    :param batch_size: 
    :param shuffle: 
    :param drop_last: 
    :param num_workers: 
    :param epochs: 
    :param model: 
    :param optimizer: 
    :param lr: 
    :param loss_fn: 
    :param epoch_start: 
    :param log_dir: 
    :param record_step: 
    :param filepath_checkpoint: 
    :return: 
    """
    # # device_ids_
    device_ids = list(range(gpu_num))
    # # Divide dataset
    t_s = datetime.now()
    dataset = MyDataset(filepath, lead)
    indices = list(range(len(dataset)))
    break_point = int(len(dataset) * break_percent)
    indices_train, indices_test = indices[: break_point], indices[break_point:]
    dataset_train = Subset(dataset, indices_train)
    dataset_test = Subset(dataset, indices_test)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=5 * gpu_num,
                                 shuffle=False, drop_last=False, num_workers=8)

    training_multiple_gpu(epochs,
                          model,
                          optimizer, lr,
                          loss_fn,
                          dataloader_train, dataloader_test,
                          epoch_start,
                          log_dir, record_step, filepath_checkpoint=filepath_checkpoint)
    t_e = datetime.now()
    print(f"耗时: {(t_e - t_s).total_seconds():.6f} 秒")


if __name__ == '__main__':
    # # Server parameter
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_num_ = torch.cuda.device_count()
    print('Num of GPU: ', gpu_num_)
    device_ids_ = list(range(gpu_num_))

    # # dataset parameters
    filepath_ = r'/dataset/northernSCS'
    # lead_ = all_varaible_stack_lead1
    lead_lst = [1, 2, 3]
    for lead_ in lead_lst:
        break_percent_ = 0.95
        batch_size_ = 5 * gpu_num_
        shuffle_ = True
        drop_last_ = True
        num_workers_ = 8

        # # model parameters
        img_size_ = 224
        in_chans_ = 161
        num_classes_ = 157

        # # log parameters
        # log_dir_ = './result/logs/test_mg'
        log_dir_ = os.path.join('result/logs', 'lead' + str(lead_))
        record_step_ = 1  # one batch per step

        # # start parameters
        start_type = 'warm'  # 'cold' or 'warm'
        filepath_checkpoint_past = os.path.join(r'result/checkpoint', 'lead' + str(lead_) + '.pth')

        # # train parameters
        epochs_ = 100
        model_ = SwinTransformerSys(img_size=img_size_, in_chans=in_chans_, num_classes=num_classes_)
        # Select the device_main_
        model_ = torch.nn.DataParallel(model_, device_ids=device_ids_)
        model_ = model_.cuda(device=device_ids_[0])
        lr_ = 0.001

        optimizer_ = torch.optim.Adam(model_.parameters(), lr=lr_)
        loss_fn_ = MSE()
        if start_type == 'warm':
            checkpoint = load_checkpoint(filepath_checkpoint_past, model_, optimizer_, device_)
            epoch_start_ = checkpoint['epoch_start']

        elif start_type == 'cold':
            delete_all_contents(log_dir_)
            epoch_start_ = 0
        else:
            raise ValueError("start_type must be 'warm' or 'cold'!")

        # # save parameters
        dict_config_dataset = {'filepath': filepath_,
                               'lead': lead_,
                               'break_percent': break_percent_,
                               'batch_size': batch_size_,
                               'shuffle': shuffle_,
                               'drop_last': drop_last_,
                               'num_workers': num_workers_}
        dict_config_train = {'epochs': epochs_,
                             'model': model_.state_dict(),
                             'lr': lr_,
                             'optimizer': optimizer_,
                             'loss_fn': loss_fn_}
        dict_config_log = {'log_dir': log_dir_,
                           'record_step': record_step_}
        dict_config = {'dataset': dict_config_dataset,
                       'train': dict_config_train,
                       'log': dict_config_log}
        filepath_config_ = os.path.join(r'result/config', 'lead' + str(lead_) + '.pkl')
        filepath_checkpoint_ = os.path.join(r'result/checkpoint', 'lead' + str(lead_) + '.pth')
        with open(filepath_config_, 'wb') as f:
            pickle.dump(dict_config, f)

        # # execute
        execute(gpu_num_,
                filepath_, lead_,
                break_percent_, batch_size_, shuffle_, drop_last_, num_workers_,
                epochs_, model_, optimizer_, lr_, loss_fn_, epoch_start_,
                log_dir_, record_step_,
                filepath_checkpoint_)
