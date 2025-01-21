import pandas as pd
from tqdm import tqdm, trange
import pickle
import numpy as np
import copy
import os
import random
from sklearn.model_selection import train_test_split

random.seed(49297)

import proc_util
from proc_util.task_build import *
from proc_util.extract_cip_label import *
import sys

mimic_data_dir = '../../../Data/mimic-iv-2.2/'

min_time_period = 48


def trim_los(data):
    """Used to build time set
    """
    num_features = len(data[0])  # 最终特征 (excluding EtCO2)
    max_length = 300  # 时间戳的最大长度(48 * 60)
    a = np.zeros((len(data), num_features, max_length))
    timestamps = []

    for i in range(len(data)):

        TS = set()
        for j in range(num_features):
            for k in range(len(data[i][j])):
                TS.add(data[i][j][k][0].to_pydatetime())

        TS = list(TS)
        TS.sort()
        timestamps.append(TS)

        for j in range(len(data[i])):
            for t, v in data[i][j]:
                idx = TS.index(t.to_pydatetime())
                if idx < max_length:
                    a[i, j, idx] = v

    print("feature extraction success")
    print("value processing success ")
    return a, timestamps


def remove_missing_dim(x, M, T):
    new_x = np.zeros((len(x), len(x[0]), len(x[0][0])))
    new_M = np.zeros((len(M), len(M[0]), len(M[0][0])))
    new_T = [[] for _ in range(len(x))]

    tmp_x = x.sum(1).squeeze()  # [B 1 L]
    for b in range(len(tmp_x)):
        new_l = 0
        for l in range(len(tmp_x[b])):
            if tmp_x[b][l] > 0:
                new_x[b, :, new_l] = x[b, :, l]
                new_M[b, :, new_l] = M[b, :, l]
                # new_T[b,:,new_l] = T[b,:,l]
                new_T[b].append(T[b][l])
                new_l += 1

    return new_x, new_M, new_T


def fix_input_format(x, T):
    """Return the input in the proper format
    x: observed values
    M: masking, 0 indicates missing values
    delta: time points of observation
    """
    timestamp = 200
    num_features = 122

    M = np.zeros_like(x)
    # x[x > 500] = 0.0
    x[x < 0] = 0.0
    M[x > 0] = 1

    x, M, T = remove_missing_dim(x, M, T)

    x = x[:, :, :timestamp]
    M = M[:, :, :timestamp]

    delta = np.zeros((x.shape[0], 1, x.shape[-1]))

    ts_len = []
    for i in range(len(T)):
        for j in range(1, len(T[i])):
            if j >= timestamp:
                break
            delta[i, 0, j] = (T[i][j] - T[i][0]).total_seconds() / 3600.0
        ts_len.append(len(T[i]))

    return x, M, delta, ts_len


def preproc_xy(adm_icu_id, data_x, data_y, dataset_name, split_name):
    out_value, out_timestamps = trim_los(data_x)
    lengths = [len(sublist) for sublist in out_timestamps]

    # 计算长度的中位数
    median_length = np.median(lengths)

    # 计算长度的最大值和最小值
    max_length = max(lengths)
    min_length = min(lengths)

    # 计算长度的平均值
    avg_length = round(sum(lengths) / len(lengths))
    print(
        f"{dataset_name} {split_name}的TS长度 - max: {max_length}, min: {min_length}, avg: {avg_length}, median: {median_length}")

    x, m, T, ts_len = fix_input_format(out_value, out_timestamps)
    # 计算长度的中位数
    lengths = ts_len
    median_length = np.median(lengths)

    # 计算长度的最大值和最小值
    max_length = max(lengths)
    min_length = min(lengths)

    # 计算长度的平均值
    avg_length = round(sum(lengths) / len(lengths))
    print(
        f"{dataset_name} {split_name}的ts_len长度 - max: {max_length}, min: {min_length}, avg: {avg_length}, median: {median_length}")
    print("timestamps format processing success")

    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)

    pickle.dump(adm_icu_id, open(dataset_name + split_name + '_sub_adm_icu_idx.p', 'wb'))
    # pickle.dump(ts_len, open(dataset_name + split_name + '_ts_len.p', 'wb'))
    save_xy(x, m, T, data_y, dataset_name, split_name)


def save_xy(in_x, in_m, in_T, label, dataset_name, split_name):
    in_T = np.expand_dims(in_T[:, 0, :], axis=1)
    x = np.concatenate((in_x, in_m, in_T), axis=1)  # input format
    y = np.array(label)
    np.save(dataset_name + split_name + '_input.npy', x)
    np.save(dataset_name + split_name + '_output.npy', y)
    print(x.shape)
    print(y.shape)

    print(dataset_name + split_name, " saved success")


def preproc_interv_xy(adm_icu_id, data_x, vent_label, vaso_label, dataset_name, split_name):
    out_value, out_timestamps = trim_los(data_x)
    lengths = [len(sublist) for sublist in out_timestamps]

    # 计算长度的中位数
    median_length = np.median(lengths)

    # 计算长度的最大值和最小值
    max_length = max(lengths)
    min_length = min(lengths)

    # 计算长度的平均值
    avg_length = sum(lengths) / len(lengths)

    # 打印
    print(
        f"{dataset_name} {split_name}的TS长度 - max: {max_length}, min: {min_length}, avg: {avg_length}, median: {median_length}")

    x, m, T, ts_len = fix_input_format(out_value, out_timestamps)
    print("timestamps format processing success")

    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)

    # pickle.dump(ts_len, open(dataset_name + split_name + '_ts_len.p', 'wb'))
    pickle.dump(adm_icu_id, open(dataset_name + split_name + '_sub_adm_icu_idx.p', 'wb'))
    save_interv_xy(x, m, T, vent_label, vaso_label, dataset_name + split_name)


def save_interv_xy(in_x, in_m, in_T, vent_label, vaso_label, save_path):
    in_T = np.expand_dims(in_T[:, 0, :], axis=1)
    x = np.concatenate((in_x, in_m, in_T), axis=1)  # input format
    vent_label = np.array(vent_label)
    vaso_label = np.array(vaso_label)

    np.save(save_path + '_input.npy', x)
    np.save(save_path + '_vent_output.npy', vent_label)
    np.save(save_path + '_vaso_output.npy', vaso_label)
    print(x.shape)
    print(vent_label.shape)
    print(vaso_label.shape)
    print(save_path, " saved success")


def create_map(icu, events):
    '''
        icu_dict: (hadm_id:(icustay_id:(intime,outtime)))
        los_dict: (hadm_id_icustay_id:los)
        adm2subj_dict: (hadm_id:subject_id)
        adm2deathtime_dict: (hadm_id:deathtime)
        feature_map: (feature:idx) #每个特征分配一个独立编号
        chart_label_dict: (feature:idx) #为每个来源于lab，chart(除机械通气外)特征分配一个独立编号
    '''
    chart_label_dict = {}
    icu_dict = {}
    los_dict = {}
    adm2subj_dict = {}
    adm2deathtime_dict = {}

    for _, p_row in tqdm(icu.iterrows(), total=icu.shape[0]):
        if p_row.HADM_ID not in icu_dict:
            icu_dict.update({p_row.HADM_ID: {p_row.ICUSTAY_ID: [p_row.INTIME, p_row.OUTTIME]}})
            los_dict.update({str(p_row.HADM_ID) + '_' + str(p_row.ICUSTAY_ID): p_row.LOS})

        elif p_row.ICUSTAY_ID not in icu_dict[p_row.HADM_ID]:
            icu_dict[p_row.HADM_ID].update({p_row.ICUSTAY_ID: [p_row.INTIME, p_row.OUTTIME]})
            los_dict.update({str(p_row.HADM_ID) + '_' + str(p_row.ICUSTAY_ID): p_row.LOS})

        if p_row.HADM_ID not in adm2subj_dict:
            adm2subj_dict.update({p_row.HADM_ID: p_row.SUBJECT_ID})

    for _, p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        if p_row.HADM_ID not in adm2deathtime_dict:
            adm2deathtime_dict.update({p_row.HADM_ID: p_row.DEATHTIME})

    # get feature set
    feature_set = []
    feature_map = {}
    events = events.loc[~(events.CHARTTIME.isna() & events.VALUENUM.isna())]

    idx = 0
    for i in events.NAME:
        if i not in feature_set:
            feature_map[i] = idx
            idx += 1
            feature_set.append(i)

    type_dict = {}
    for i in feature_set:
        tmp_p = events.loc[events.NAME.isin([i])]
        tmp_set = set(tmp_p.TABLE)
        type_dict.update({i: tmp_set})

    idx = 0
    for k in type_dict:
        if 'chart' in type_dict[k] or 'lab' in type_dict[k]:
            if k not in chart_label_dict and k != "Mechanical Ventilation":
                chart_label_dict[k] = idx
                idx += 1

    print("got ", str(len(feature_set)), " features")
    return feature_map, chart_label_dict, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict


if __name__ == '__main__':
    data_root_folder = "../../../Data/mimic-iv-2.2/"

    if not os.path.isdir(data_root_folder):
        os.mkdir(data_root_folder)

    data_tmp_folder = data_root_folder + "tmp/"

    if not os.path.isdir(data_tmp_folder):
        os.mkdir(data_tmp_folder)

    adm_id_folder = "./adm_id/"

    bio_path = data_tmp_folder + "patient_records_large.p"
    interv_outPath = data_tmp_folder + "all_hourly_data.h5"
    resource_path = "./proc_util/resource/"

    print("Loading data...")
    icu = pd.read_csv(mimic_data_dir + 'icustays.csv.gz',
                      usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'])
    icu.columns = icu.columns.str.upper()
    icu.rename(columns={'STAY_ID': 'ICUSTAY_ID'}, inplace=True)
    icu.drop_duplicates(inplace=True)

    adm = pd.read_csv(mimic_data_dir + 'admissions.csv.gz',
                      usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'])
    adm.columns = adm.columns.str.upper()
    adm.rename(columns={'STAY_ID': 'ICUSTAY_ID'}, inplace=True)
    adm.drop_duplicates(inplace=True)

    mimi_iv_event = './mimic_ards_events.csv.gz'

    events = pd.read_csv(mimi_iv_event, usecols=['HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'VALUENUM', 'TABLE', 'NAME'])
    events.drop_duplicates(inplace=True)

    events = events.loc[~(events.CHARTTIME.isna() & events.VALUENUM.isna())]

    feature_map, chart_label_dict, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict = create_map(icu, events)

    # 剔除一部分特征
    remove_mod_idx = [8, 11, 18, 23, 30, 31, 36, 38, 39, 42, 45, 46, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                      70, 71, 72, 73, 74, 75, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                      97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117,
                      118, 119]

    tmp_feature_name = sorted(feature_map.items(), key=lambda d: d[1], reverse=False)

    feature_map = {}
    feature_name = []
    new_idx = 0
    for i, j in tmp_feature_name:
        if j not in remove_mod_idx:
            feature_map[i] = new_idx
            new_idx += 1
            feature_name.append(i)
    print("got ", str(len(feature_map)), " features")

    tmp_feature_name = sorted(feature_map.items(), key=lambda d: d[1], reverse=False)
    df = pd.DataFrame(tmp_feature_name, columns=['Feature Name', 'Value'])
    df.to_csv('feature_mapping_mimicv.csv', index=False)
    chart_label_dict = feature_map


    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    adm.ADMITTIME = pd.to_datetime(adm.ADMITTIME)
    adm.DISCHTIME = pd.to_datetime(adm.DISCHTIME)
    adm.DEATHTIME = pd.to_datetime(adm.DEATHTIME)

    '''
    mimi_iv_icu = './mimic_ards_icu.csv.gz'
    mimi_iv = pd.read_csv(mimi_iv_icu, usecols=['HADM_ID'])
    mimi_iv.drop_duplicates(inplace=True)
    mimi_iv = mimi_iv['HADM_ID'].tolist()
    train_val_set, test_set = train_test_split(mimi_iv, train_size=0.8, random_state=42, shuffle=True)
    train_set, val_set = train_test_split(train_val_set, train_size=0.8, random_state=42, shuffle=False)
    with open(adm_id_folder + 'train_adm_idx.p', 'wb') as f:
        pickle.dump(train_set, f)
    with open(adm_id_folder + 'test_adm_idx.p', 'wb') as f:
        pickle.dump(test_set, f)
    with open(adm_id_folder + 'val_adm_idx.p', 'wb') as f:
        pickle.dump(val_set, f)
    '''
    train_adm_id = pickle.load(open(adm_id_folder + 'train_adm_idx.p', 'rb'))
    test_adm_id = pickle.load(open(adm_id_folder + 'test_adm_idx.p', 'rb'))
    val_adm_id = pickle.load(open(adm_id_folder + 'val_adm_idx.p', 'rb'))
    # #==== Decompensation ====icu滚动
    print("Building Decompensation task...")

    adm_icu_id, decom_data, decom_label = create_decompensation_large(adm, events, feature_map, icu_dict, los_dict,
                                                                      adm2deathtime_dict, adm2subj_dict, \
                                                                      sample_rate=12, shortest_length=24,
                                                                      future_time_interval=24.0,
                                                                      filt_adm_ids=train_adm_id)

    preproc_xy(adm_icu_id, decom_data, decom_label, data_root_folder + 'decom/', 'train')

    adm_icu_id, decom_data, decom_label = create_decompensation_large(adm, events, feature_map, icu_dict, los_dict,
                                                                      adm2deathtime_dict, adm2subj_dict, \
                                                                      sample_rate=12, shortest_length=24,
                                                                      future_time_interval=24.0,
                                                                      filt_adm_ids=test_adm_id)

    preproc_xy(adm_icu_id, decom_data, decom_label, data_root_folder + 'decom/', 'test')

    adm_icu_id, decom_data, decom_label = create_decompensation_large(adm, events, feature_map, icu_dict, los_dict,
                                                                      adm2deathtime_dict, adm2subj_dict, \
                                                                      sample_rate=12, shortest_length=24,
                                                                      future_time_interval=24.0,
                                                                      filt_adm_ids=val_adm_id)

    preproc_xy(adm_icu_id, decom_data, decom_label, data_root_folder + 'decom/', 'val')

    print("Build Decompensation task done")