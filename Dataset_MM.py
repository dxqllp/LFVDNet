from math import inf
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn import model_selection
import pandas as pd
from tqdm import tqdm
from data_utils.physionet import PhysioNet
from functools import partial

Constants_PAD = 0


class NpyDataset(Dataset):
    def __init__(self, file_path_x, file_path_y):
        self.data_x = np.load(file_path_x, mmap_mode='r', allow_pickle=True)
        self.data_y = np.load(file_path_y, mmap_mode='r', allow_pickle=True)
        self.num_samples = self.data_x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_x = self.data_x[idx]
        sample_y = self.data_y[idx]
        return sample_x, sample_y


def get_data_min_max(records, device):
    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)
    data_len = []

    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1)
        data_len.append(len(tt))
        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals).to(device))
                batch_max.append(torch.max(non_missing_vals).to(device))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)
    data_len = torch.tensor(data_len)
    print('序列长度', '最大值', torch.max(data_len), '中位数', torch.median(data_len), '最小值', torch.min(data_len))

    return data_min, data_max


def proc_hii_data(batch, input_dim, args, stats_path):
    x = np.array([data for data, label in batch])
    y = np.array([label for data, label in batch])
    x = x[:, :input_dim * 2 + 1]
    x[:, -1, :] = x[:, -1, :] / 24

    if args.debug_flag:
        x = x[:1000, :]
        y = y[:1000]

    if args.task == "los":
        y = y - 1

    x = np.transpose(x, (0, 2, 1))

    new_x = np.zeros((len(x), len(x[0]), input_dim * 3 + 1))

    total = len(x)
    batch_sz = 20000

    pbar = range(0, total, batch_sz)
    for start in pbar:
        end = min(start + batch_sz, total)

        new_x[start:end, :, :input_dim * 2 + 1] = process_data(x[start:end], input_dim, task=args.task,
                                                               stats_path=stats_path)
        new_x[start:end, :, input_dim * 2 + 1:input_dim * 3 + 1] = cal_tau(x[start:end, :, -1],
                                                                           x[start:end, :, input_dim:2 * input_dim])
    new_x = torch.from_numpy(new_x).float()
    y = torch.from_numpy(y).long().squeeze()
    batch = (new_x, y)
    return batch

def get_clints_hii_data(args, to_set=False):
    data_folder_x = args.root_path + args.data_path + args.task + '/'
    data_folder_y = args.root_path + args.data_path + args.task + '/'
    dataloader = []

    for set_name in ['train', 'val', 'test']:
        data_x_all = []
        data_y_all = []

        if set_name == 'train':
            shuffle = True
        else:
            shuffle = False

        print("loading " + set_name + " data")

        data_x_all = np.load(data_folder_x + set_name + '_input.npy', allow_pickle=True, mmap_mode='r')

        args.num_types = int((data_x_all.shape[1] - 1) / 2)
        dataset = NpyDataset(file_path_x=data_folder_x + set_name + '_input.npy',
                             file_path_y=data_folder_y + set_name + '_output.npy')
        collate_fn = partial(proc_hii_data, input_dim=args.num_types, args=args, stats_path=data_folder_x + set_name)
        if set_name == 'train':
            train_generator = torch.Generator()
            train_generator.manual_seed(8)
            dataloader_ = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=1,
                collate_fn=collate_fn, generator=train_generator)
        else:
            dataloader_ = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=1,
                collate_fn=collate_fn)
        dataloader.append(dataloader_)
        del data_x_all, data_y_all
    print("type num: ", args.num_types)
    return dataloader[0], dataloader[1], dataloader[2]


def cal_tau(observed_tp, observed_mask):
    # input [B,L,K], [B,L]
    # return [B,L,K]
    # observed_mask, observed_tp = x[:, :, input_dim:2 * input_dim], x[:, :, -1]
    if observed_tp.ndim == 2:
        tmp_time = observed_mask * np.expand_dims(observed_tp, axis=-1)  # [B,L,K]
    else:
        tmp_time = observed_tp.copy()

    b, l, k = tmp_time.shape

    new_mask = observed_mask.copy()
    new_mask[:, 0, :] = 1
    tmp_time[new_mask == 0] = np.nan
    tmp_time = tmp_time.transpose((1, 0, 2))  # [L,B,K]
    tmp_time = np.reshape(tmp_time, (l, b * k))  # [L, B*K]

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.fillna(method='ffill')
    tmp_time = np.array(df1)

    tmp_time = np.reshape(tmp_time, (l, b, k))
    tmp_time = tmp_time.transpose((1, 0, 2))  # [B,L,K]

    tmp_time[:, 1:] -= tmp_time[:, :-1]
    del new_mask
    return tmp_time * observed_mask


def process_data(x, input_dim, task, stats_path=None, m=None, tt=None, x_only=False):
    if not x_only:
        observed_vals, observed_mask, observed_tp = x[:, :,
                                                    :input_dim], x[:, :, input_dim:2 * input_dim], x[:, :, -1]
        observed_tp = np.expand_dims(observed_tp, axis=-1)
    else:
        observed_vals = x
        assert m is not None
        observed_mask = m
        observed_tp = tt
    if task == 'decom':
        observed_vals = tensorize_normalize(observed_vals, task, stats_path=stats_path)
    else:
        observed_vals = tensorize_normalize(observed_vals, task)
    observed_vals[observed_mask == 0] = 0
    if not x_only:
        return np.concatenate((observed_vals, observed_mask, observed_tp), axis=-1)
    return observed_vals


def tensorize_normalize(P_tensor, task, stats_path=None):
    if task == 'decom':
        mf = np.load(stats_path + '_mf.npy')
        stdf = np.load(stats_path + '_stdf.npy')
        P_tensor = normalize(P_tensor, mf, stdf)
    else:
        mf, stdf = getStats(P_tensor)
        P_tensor = normalize(P_tensor, mf, stdf)
    return P_tensor


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        if len(vals_f) > 0:
            mf[f] = np.mean(vals_f)
            tmp_std = np.std(vals_f)
            stdf[f] = np.max([tmp_std, eps])
    return mf, stdf


def normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    return Pnorm_tensor


def variable_time_collate_fn(batch, device, input_dim, task, return_np=False, to_set=False, maxlen=200,
                             data_min=None, data_max=None, activity=False):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    # number of labels
    # N = batch[0][-1].shape[1] if activity else 1
    if maxlen is None:
        len_tt = [ex[1].size(0) for ex in batch]
        maxlen = np.max(len_tt)

    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)

    if activity:
        combined_labels = torch.zeros([len(batch), maxlen]).to(device)
    else:
        combined_labels = torch.zeros([len(batch)]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = min(tt.size(0), maxlen)
        enc_combined_tt[b, :currlen] = tt[:currlen].to(device)
        enc_combined_vals[b, :currlen] = vals[:currlen].to(device)
        enc_combined_mask[b, :currlen] = mask[:currlen].to(device)

        if labels.dim() == 2:
            combined_labels[b] = torch.argmax(labels, dim=-1)
        else:
            combined_labels[b] = labels.to(device)

    enc_combined_vals = torch.tensor(
        process_data(enc_combined_vals.cpu().numpy(), m=enc_combined_mask.cpu().numpy(), tt=enc_combined_tt,
                     input_dim=input_dim, x_only=True, task=task)).to(enc_combined_tt.device)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    tau = torch.tensor(cal_tau(enc_combined_tt.cpu().numpy(), enc_combined_mask.cpu().numpy())).to(
        enc_combined_vals.device)
    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1), tau), 2)

    return combined_data, combined_labels

def get_physionet_data(args, device, q=0.016, flag=1, set=4000):
    train_dataset_obj = PhysioNet(args.data_path + 'physionet', train=True,
                                  quantization=q,
                                  download=True,
                                  device=device, set=set)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]
    data_min, data_max = get_data_min_max(total_dataset, device)
    print(len(total_dataset))
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    input_dim = vals.size(-1)
    batch_size = min(len(train_dataset_obj), args.batch_size)
    args.num_types = input_dim

    if not args.retrain:
        train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                random_state=11, shuffle=False)

        val_data_combined = variable_time_collate_fn(val_data, device, input_dim=input_dim, data_min=data_min,
                                                     data_max=data_max, task=args.task)

        val_data_combined = TensorDataset(
            val_data_combined[0], val_data_combined[1].long().squeeze())

        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None

    train_data_combined = variable_time_collate_fn(train_data, device, input_dim=input_dim, data_min=data_min,
                                                   data_max=data_max, task=args.task)
    test_data_combined = variable_time_collate_fn(test_data, device, input_dim=input_dim, data_min=data_min,
                                                  data_max=data_max, task=args.task)

    # norm_mean = train_data_combined[0][:, :, :input_dim].mean(dim=0, keepdim=True).cpu()

    print(train_data_combined[1].sum(
    ), test_data_combined[1].sum())
    print(train_data_combined[0].size(), train_data_combined[1].size(),
          test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long().squeeze())

    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long().squeeze())

    train_generator = torch.Generator()
    train_generator.manual_seed(8)  # 这个生成器依然是独立的

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True, generator=train_generator)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, input_dim

