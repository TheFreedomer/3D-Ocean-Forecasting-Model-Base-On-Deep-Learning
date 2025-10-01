import os.path
import pickle
import torch
from package.utils.MyDataset import MyDataset
from package.utils.utils import training_single_gpu, load_checkpoint, delete_all_contents
from package.utils.Cost import MSE
from torch.utils.data import Subset, DataLoader
from datetime import datetime
from package.SwinUnet_main.networks.swin_transformer_unet_skip_expand_decoder_sys import \
    SwinTransformerSys


def execute(device,
            filepath, lead,
            break_percent, batch_size, shuffle, drop_last, num_workers,
            epochs, model, optimizer, lr, loss_fn, epoch_start,
            log_dir, record_step,
            filepath_checkpoint):
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
    dataloader_test = DataLoader(dataset_test, batch_size=1,
                                 shuffle=False, drop_last=False, num_workers=0)
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
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_type = 'warm'     # 'cold' or 'warm'
    filepath_checkpoint_past = r'../result/checkpoint/measure.pth'

    # # dataset parameters
    filepath_ = r'/dataset/northernSCS'
    lead_ = 1
    break_percent_ = 0.95
    batch_size_ = 5
    shuffle_ = True
    drop_last_ = True
    num_workers_ = 1

    # # model parameters
    img_size_ = 224
    in_chans_ = 161
    num_classes_ = 157

    # # log parameters
    log_dir_ = '../result/logs/measure'
    record_step_ = 1  # one batch per step

    # # train parameters
    epochs_ = 3
    model_ = SwinTransformerSys(img_size=img_size_, in_chans=in_chans_, num_classes=num_classes_)
    model_.to(device_)
    lr_ = 0.1
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=lr_)
    # loss_fn_ = torch.nn.MSELoss()
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
    filepath_config_ = os.path.join(r'result/config', 'measure.pkl')
    filepath_checkpoint_ = os.path.join(r'result/checkpoint', 'measure.pth')
    with open(filepath_config_, 'wb') as f:
        pickle.dump(dict_config, f)

    # # execute
    execute(device_,
            filepath_, lead_,
            break_percent_, batch_size_, shuffle_, drop_last_, num_workers_,
            epochs_, model_, optimizer_, lr_, loss_fn_, epoch_start_,
            log_dir_, record_step_,
            filepath_checkpoint_)
