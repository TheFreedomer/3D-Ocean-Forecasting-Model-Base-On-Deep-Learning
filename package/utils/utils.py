import torch
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from tqdm import tqdm


def training_single_gpu(device,
                        epochs,
                        model,
                        optimizer, lr,
                        loss_fn,
                        dataloader_train, dataloader_test,
                        epoch_start=0,
                        log_dir='./result/logs/measure', record_step=10,
                        filepath_checkpoint='./result/checkpoint/measure.pth',
                        text_display=False):
    """
    Single GPU and batch train
    :param device:
    :param epochs:
    :param model:
    :param optimizer:
    :param lr:
    :param loss_fn:
    :param dataloader_train:
    :param dataloader_test:
    :param epoch_start:
    :param log_dir:
    :param record_step:
    :param filepath_checkpoint:
    :param text_display:
    :return:
    """
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (feature_train, label_train) in enumerate(tqdm(dataloader_train)):
            # 数据移动到device
            feature_train = feature_train.to(device)
            label_train = label_train.to(device)

            predict_train = model(feature_train)
            loss_train = loss_fn(predict_train, label_train)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        if epoch_start % record_step == 0:
            with torch.no_grad():
                loss_sum = 0
                n = 0
                for feature_test, label_test in dataloader_test:
                    feature_test = feature_test.to(device)
                    label_test = label_test.to(device)
                    model.eval()
                    predict_test = model(feature_test)
                    model.train()
                    loss_test = loss_fn(predict_test, label_test)
                    n_current = label_test.shape[0]
                    loss_sum += loss_test * n_current
                    n += n_current
                loss_test = loss_sum / n
                writer.add_scalar('Loss of Training set', loss_train, epoch_start)
                writer.add_scalar('Loss of Test set', loss_test, epoch_start)
                if text_display:
                    print("Epoch: {}, "
                          "Loss train: {}, "
                          "Loss measure: {}".format(epoch_start, loss_train, loss_test))
        adjust_learning_rate_exp(optimizer, epoch_start, lr)
        epoch_start += 1

    save_checkpoint(filepath_checkpoint, model, optimizer, epoch_start)


def training_multiple_gpu(device_main,
                          epochs,
                          model,
                          optimizer, lr,
                          loss_fn,
                          dataloader_train, dataloader_test,
                          epoch_start=0,
                          log_dir='./result/logs/measure', record_step=10,
                          filepath_checkpoint='./result/checkpoint/measure.pth',
                          text_display=False):
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (feature_train, label_train) in enumerate(tqdm(dataloader_train)):
            # 数据移动到device
            feature_train = feature_train.cuda(device=device_main)
            label_train = label_train.cuda(device=device_main)

            predict_train = model(feature_train)
            # loss_train = loss_fn(predict_train[:, :, :, 0: 169, :], label_train[:, :, :, 0: 169, :])
            loss_train = loss_fn(predict_train, label_train)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        if epoch_start % record_step == 0:
            with torch.no_grad():
                loss_sum = 0
                n = 0
                for feature_test, label_test in dataloader_test:
                    feature_test = feature_test.cuda(device=device_main)
                    label_test = label_test.cuda(device=device_main)
                    model.eval()
                    predict_test = model(feature_test)
                    model.train()
                    loss_test = loss_fn(predict_test, label_test).item()
                    n_current = label_test.shape[0]
                    loss_sum += loss_test * n_current
                    n += n_current
                loss_test = loss_sum / n
                writer.add_scalar('Loss of Training set', loss_train, epoch_start)
                writer.add_scalar('Loss of Test set', loss_test, epoch_start)
                if text_display:
                    print("Epoch: {}, "
                          "Loss train: {}, "
                          "Loss measure: {}".format(epoch_start, loss_train, loss_test))
        adjust_learning_rate_exp(optimizer, epoch, lr)

        epoch_start += 1

        save_checkpoint(filepath_checkpoint, model, optimizer, epoch_start)


def adjust_learning_rate_exp(optimizer, epoch, init_lr=0.1, decay_rate=0.95):
    """
    The learning rate decays exponentially in each epoch
    :param optimizer:
    :param epoch:
    :param init_lr:
    :param decay_rate:
    :return:
    """
    lr = init_lr * (decay_rate ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(filepath_checkpoint, model, optimizer, epoch_start):
    """

    :param filepath_checkpoint:
    :param model:
    :param optimizer:
    :param epoch_start:
    :return:
    """
    checkpoint = {
        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel)
        else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_start': epoch_start
    }
    torch.save(checkpoint, filepath_checkpoint)


def load_checkpoint(filepath_checkpoint, model, optimizer, device):
    """

    :param filepath_checkpoint:
    :param model:
    :param optimizer:
    :param device:
    :return:
    """
    checkpoint = torch.load(filepath_checkpoint, weights_only=True)
    model_state_dict = checkpoint['model_state_dict']
    if isinstance(model, torch.nn.DataParallel):
        # IF Multiple GPU
        if not any(key.startswith('module.') for key in model_state_dict.keys()):
            model_state_dict = {'module.' + key: val for key, val in model_state_dict.items()}
    else:
        # IF Single GPU
        model_state_dict = {key.replace('module.', ''): val for key, val in model_state_dict.items()}
    # Load state dict
    model.load_state_dict(model_state_dict)
    # Load state optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Optimizer to device_main_
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # for param_group in optimizer.param_groups:
    #     for param in param_group["params"]:
    #         param.data = param.data.to(device)
    #         if param.grad is not None:
    #             param.grad.data = param.grad.data.to(device)

    return checkpoint


def create_directory(directory_path):
    """

    :param directory_path:
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Directory has created：", directory_path)
    else:
        print("Directory is existed：", directory_path)


def delete_all_contents(folder_path):
    """
    Delete all the contents under the folder (including subdirectories and files).
    :param folder_path:
    :return:
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f"The folder has been emptied: {folder_path}")
    else:
        print(f"The path does not exist: {folder_path}")

