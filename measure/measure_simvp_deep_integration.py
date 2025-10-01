import torch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from package.SimVP_Simpler_yet_Better_Video_Prediction_master.model import SimVP

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

lead_lst = [1, 2, 3]
for lead in lead_lst:
    dir_sub = 'deep_integration_lead' + str(lead)
    dir_sub_patent = 'deep_integration_lead_patent' + str(lead)
    filepath_config = os.path.join(r'../result/config', 'deep_integration_' + str(lead) + '.pkl')
    with open(filepath_config, 'rb') as f:
        config = pickle.load(f)
    filepath_checkpoint = r'../result/checkpoint/deep_integration_'+ str(lead) + '.pth'

    device_ids = [3]
    gpu_num = len(device_ids)

    shape_in_ = config['model']['shape_in']
    seq_len_, channel_in_, H_, W_ = shape_in_
    channel_out_ = config['model']['channel_out']
    hid_S_ = config['model']['hid_S']
    hid_T_ = config['model']['hid_T']
    N_S_ = config['model']['N_S']
    N_T_ = config['model']['N_T']
    groups_ = config['model']['groups']

    model = SimVP(shape_in=shape_in_, channel_out=channel_out_,
                   hid_S=hid_S_, hid_T=hid_T_, N_S=N_S_, N_T=N_T_,
                   groups=groups_)
    model = model.cuda(device=device_ids[0])

    checkpoint = torch.load(filepath_checkpoint, weights_only=False, map_location='cuda:' + str(device_ids[0]))
    model.load_state_dict(checkpoint['model_state_dict'])

    # # build dataset
    from package.utils.MyDataset import Dataset
    from torch.utils.data import Subset, DataLoader

    def transform_x(x):
        x = np.ascontiguousarray(x)
        return x

    def transform_y(y):
        y = np.ascontiguousarray(y)
        return y

    time_axis = np.load('../data/info_axis/time.npy')
    dir_dataset = r'../data/dataset'
    filepath_statistics = r'../data/statistics/all/statistics.pkl'

    lst_variable_feature = config['dataset']['lst_variable_feature']
    lst_is_mask_feature = config['dataset']['lst_is_mask_feature']
    lst_level_feature = config['dataset']['lst_level_feature']

    lst_variable_label = config['dataset']['lst_variable_label']
    lst_is_mask_label = config['dataset']['lst_is_mask_label']
    lst_level_label = config['dataset']['lst_level_label']

    dim_level_cat = config['dataset']['dim_level_cat']
    dataset = Dataset(lead,
                      time_axis, dir_dataset, filepath_statistics,
                      lst_variable_feature, lst_is_mask_feature, lst_level_feature,
                      lst_variable_label, lst_is_mask_label, lst_level_label,
                      dim_level_cat,
                      transform_x, transform_y)

    indices = list(range(len(dataset)))
    break_point = int(len(dataset) * config['dataset']['break_percent'])
    indices_test = indices[break_point:]
    dataset_test = Subset(dataset, indices_test)
    dataloader_test = DataLoader(dataset_test, batch_size=5,
                                 shuffle=False, drop_last=False, num_workers=gpu_num)

    model.eval()
    with torch.no_grad():
        for i, (feature_test, label_test) in enumerate(dataloader_test):
            feature_test = feature_test.cuda(device=device_ids[0])
            label_test = label_test.cuda(device=device_ids[0])
            predict_test = model(feature_test)
            x = predict_test.to('cpu').detach().numpy()
            y = label_test.to('cpu').detach().numpy()
            if i == 0:
                x_merge = x
                y_merge = y
            else:
                x_merge = np.concatenate((x_merge, x), axis=0)
                y_merge = np.concatenate((y_merge, y), axis=0)

    # mask
    mask_all = np.all(y_merge == 0, axis=0)

    lon = np.load('../data/info_axis/lon_img.npy')
    lat = np.load('../data/info_axis/lat_img.npy')
    depth = np.load('../data/info_axis/depth_img.npy')
    time_axis = np.load('../data/info_axis/time.npy')[lead:][break_point:]
    variable_lst = ['u', 'v', 't', 's']
    title_lst = ['UO', 'VO', 'Temperature', 'Salinity']
    title_cn_lst = ['纬向海水流速', '经向海水流速', '海水温度', '海水盐度', ]
    unit_lst = ['m/s', 'm/s', '℃', 'PSU']

    # RMSE
    rmse = np.sqrt(np.nanmean((x_merge - y_merge) ** 2, axis=0))
    rmse[mask_all] = np.nan

    # PCC
    pcc = np.full_like(rmse, fill_value=np.nan)
    for d in range(depth.size):
        for v in range(len(variable_lst)):
            for lat_n in range(lat.size):
                for lon_n in range(lon.size):
                    if ~mask_all[d, v, lat_n, lon_n]:
                        x_temp = x_merge[:, d, v, lat_n, lon_n]
                        y_temp = y_merge[:, d, v, lat_n, lon_n]
                        pcc_temp = np.corrcoef(x_temp, y_temp)[0, 1]
                        # update
                        pcc[d, v, lat_n, lon_n] = pcc_temp

    # point target
    lon_p1, lat_p1 = 108, 20
    lon_p2, lat_p2 = 111.5, 19.5

    idx_lon1, idx_lat1 = np.argmin(np.abs(lon - lon_p1)), np.argmin(np.abs(lat - lat_p1))
    idx_lon2, idx_lat2 = np.argmin(np.abs(lon - lon_p2)), np.argmin(np.abs(lat - lat_p2))

    def get_color_range_max(arr, mask, range_target=(0.9, 0.95)):
        size_all = arr[~mask].size
        value_min = np.nanmin(arr)
        value_max = np.nanmax(arr)

        ratio_target = 0
        value_left = value_min
        value_right = value_max
        value_target = (value_left + value_right) / 2
        while ratio_target < range_target[0] or ratio_target > range_target[1]:
            ratio_target = arr[arr <= value_target].size / size_all
            if ratio_target < range_target[0]:
                value_left = value_target
            else:
                value_right = value_target
            value_target = (value_left + value_right) / 2
        return value_target, ratio_target


    def get_color_range_min(arr, mask, range_target=(0.9, 0.95)):
        size_all = arr[~mask].size
        value_min = np.nanmin(arr)
        value_max = np.nanmax(arr)

        ratio_target = 0
        value_left = value_min
        value_right = value_max
        value_target = (value_left + value_right) / 2
        while ratio_target < range_target[0] or ratio_target > range_target[1]:
            ratio_target = arr[arr >= value_target].size / size_all
            if ratio_target < range_target[0]:
                value_right = value_target
            else:
                value_left = value_target
            value_target = (value_left + value_right) / 2
        return value_target, ratio_target

    for i in range(rmse.shape[1]):
        print(variable_lst[i])
        rmse_min_temp = np.nanmin(rmse[:, i, :])
        rmse_max_temp = np.nanmax(rmse[:, i, :])
        rmse_mean_temp = np.nanmean(rmse[:, i, :])
        print(rmse_min_temp, '\t', rmse_max_temp, '\t', rmse_mean_temp)

        pcc_min_temp = np.nanmin(pcc[:, i, :])
        pcc_max_temp = np.nanmax(pcc[:, i, :])
        pcc_mean_temp = np.nanmean(pcc[:, i, :])
        print(pcc_min_temp, '\t', pcc_max_temp, '\t', pcc_mean_temp)

    variable_s_lst = ['u', 'u', 'v', 'v', 't', 't', 's', 's', ]
    depth_s_lst = [0, 50, 0, 50, 0, 50, 0, 50, ]
    # variable_s_lst = ['u']
    # depth_s_lst = [0]

    for i in range(len(variable_s_lst)):
        variable_s = variable_s_lst[i]
        depth_s = depth_s_lst[i]

        idx_depth = np.argmin(np.abs(depth - depth_s))
        idx_variable = variable_lst.index(variable_s)

        depth_str = str(round(depth[idx_depth], 2))

        lon_axis_temp = lon
        lat_axis_temp = lat
        time_axis_temp = time_axis.astype('int64') / 10**9

        rmse_temp = rmse[idx_depth, idx_variable, :]
        pcc_temp = pcc[idx_depth, idx_variable, :]

        # mask rmse
        mask_temp = mask_all[idx_depth, idx_variable, :]
        rmse_temp[mask_temp] = np.nan

        # get color range
        rmse_min_color, ratio_rmse_min = get_color_range_min(rmse_temp, mask_temp)
        rmse_max_color, ratio_rmse_max = get_color_range_max(rmse_temp, mask_temp)

        pcc_min_color, ratio_pcc_min = get_color_range_min(pcc_temp, mask_temp)
        pcc_max_color, ratio_pcc_max = get_color_range_max(pcc_temp, mask_temp)

        # clean nan
        rmse_temp = rmse_temp[np.isfinite(rmse_temp)].reshape(-1)
        pcc_temp = pcc_temp[np.isfinite(pcc_temp)].reshape(-1)

        #
        lon_point_temp1 = lon[idx_lon1]
        lat_point_temp1 = lat[idx_lat1]

        lon_point_temp2 = lon[idx_lon2]
        lat_point_temp2 = lat[idx_lat2]

        prediction_point_temp1 = x_merge[:, idx_depth, idx_variable, idx_lat1, idx_lon1]
        label_point_temp1 = y_merge[:, idx_depth, idx_variable, idx_lat1, idx_lon1]

        prediction_point_temp2 = x_merge[:, idx_depth, idx_variable, idx_lat2, idx_lon2]
        label_point_temp2 = y_merge[:, idx_depth, idx_variable, idx_lat2, idx_lon2]

        fontsize = 18
        s_size = 50
        # 创建图形和子图
        fig = plt.figure(figsize=(22, 18), facecolor='white')
        # 主网格
        gs = fig.add_gridspec(2, 2)

        # ==========  (1, 1) ==========
        ax1 = fig.add_subplot(gs[0])
        bins = np.linspace(rmse_min_color, rmse_max_color, 7)
        counts, bins, patches = ax1.hist(rmse_temp, bins=bins,
                                       edgecolor='white',
                                       color='k',
                                       alpha=1)
        for count, patch in zip(counts, patches):
            if count > 0:  # 只标注有数据的柱子
                ax1.text(patch.get_x() + patch.get_width()/2,  # x位置：柱子中心
                        patch.get_height() + 0.5,             # y位置：柱子高度上方
                        f'{int(count)}',                      # 显示频数（取整）
                        ha='center',                          # 水平居中
                        va='bottom',                          # 垂直底部对齐
                        fontsize=fontsize-2)                  # 比主字体稍小
        ax1.set_xticks(bins)
        ax1.set_xticklabels([f'{x:.3f}' for x in bins], rotation=45)  # 保留2位小数并旋转45度防重叠

        title = (f"a. {title_cn_lst[idx_variable]}RMSE频数分布"
                 f"(水深：{depth_str}m)")
        ax1.set_title(title, fontsize=fontsize + 2, pad=15, fontweight='bold')
        ax1.set_xlabel('RMSE数值区间' + '(' + unit_lst[idx_variable] + ')', fontsize=fontsize + 2)
        ax1.set_ylabel('频数', fontsize=fontsize + 2)

        # 调整刻度标签大小
        ax1.tick_params(labelsize=fontsize)

        # ========== (2, 1) ==========
        ax3 = fig.add_subplot(gs[2])
        bins = np.linspace(pcc_min_color, pcc_max_color, 7)
        counts, bins, patches = ax3.hist(pcc_temp, bins=bins,
                                       edgecolor='white',
                                       color='k',
                                       alpha=1)
        for count, patch in zip(counts, patches):
            if count > 0:  # 只标注有数据的柱子
                ax3.text(patch.get_x() + patch.get_width()/2,  # x位置：柱子中心
                        patch.get_height() + 0.5,             # y位置：柱子高度上方
                        f'{int(count)}',                      # 显示频数（取整）
                        ha='center',                          # 水平居中
                        va='bottom',                          # 垂直底部对齐
                        fontsize=fontsize-2)                  # 比主字体稍小
        ax3.set_xticks(bins)
        ax3.set_xticklabels([f'{x:.3f}' for x in bins], rotation=45)  # 保留2位小数并旋转45度防重叠

        title = (f"c. {title_cn_lst[idx_variable]}PCC频数分布"
                 f"(水深：{depth_str}m)")
        ax3.set_title(title, fontsize=fontsize + 2, pad=15, fontweight='bold')
        ax3.set_xlabel('PCC数值区间', fontsize=fontsize + 2)
        ax3.set_ylabel('频数', fontsize=fontsize + 2)

        # 调整刻度标签大小
        ax3.tick_params(labelsize=fontsize)

        # ==========（1, 2）==========
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(time_axis, label_point_temp1, label='真实值',
                         color='k', linestyle='-', linewidth=2)
        ax2.plot(time_axis, prediction_point_temp1, label='预报值',
                         color='k', linewidth=3, linestyle='--')

        ax2.set_title(f'b. 性能展示点A\n({lon_point_temp1:.2f}°E, {lat_point_temp1:.2f}°N)',
                     fontsize=fontsize + 2, pad=15, fontweight='bold')

        # 设置时间序列图的刻度标签大小
        ax2.tick_params(axis='both', which='major', labelsize=fontsize-4)

        ax2.set_ylabel(title_cn_lst[idx_variable] + ' (' + unit_lst[idx_variable] + ')', fontsize=fontsize)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=fontsize, framealpha=0.8)
        # 调整刻度标签大小
        ax2.tick_params(labelsize=fontsize)

        # ==========（2, 2）==========
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(time_axis, label_point_temp2, label='真实值',
                         color='k', linestyle='-', linewidth=2)
        ax4.plot(time_axis, prediction_point_temp2, label='预报值',
                         color='k', linewidth=3, linestyle='--')

        ax4.set_title(f'd. 性能展示点B\n({lon_point_temp2:.2f}°E, {lat_point_temp2:.2f}°N)',
                     fontsize=fontsize + 2, pad=15, fontweight='bold')

        # 设置时间序列图的刻度标签大小
        ax4.tick_params(axis='both', which='major', labelsize=fontsize-4)

        ax4.set_ylabel(title_cn_lst[idx_variable] + ' (' + unit_lst[idx_variable] + ')', fontsize=fontsize)
        ax4.grid(True, linestyle='--', alpha=0.6)
        ax4.legend(fontsize=fontsize, framealpha=0.8)
        # 调整刻度标签大小
        ax4.tick_params(labelsize=fontsize)

        # 调整布局
        plt.tight_layout()
        filename_save = 'lead' + str(lead) + '-' + variable_s + '-' + 'depth' + depth_str + 'm.png'
        filepath_save = os.path.join('../data', 'data_draw', dir_sub, filename_save)
        plt.savefig(filepath_save, dpi=300)
        plt.close()