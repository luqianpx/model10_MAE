import os
import torch
import numpy as np
from torch.utils.data import Dataset

# dataset
def ECG_dataset(args):
    ecg_da, ecg_la = read_ECG(args)

    # Check if the labels are valid (non-negative)
    if np.max(ecg_la) < 0:
        raise ValueError("Maximum value in ecg_la is negative, which is unexpected.")

    if args.dataset_type == 'pretrain':
        train_dataset = CustomTensorDataset(data=(ecg_da, ecg_la))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_dataset = CustomTensorDataset(data=(ecg_da, ecg_la))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.logistic_batch_size, shuffle=True)

    # update args
    args.n_channel = ecg_da.shape[1]
    args.n_length = ecg_da.shape[2]
    args.n_class = int(np.max(ecg_la) + 1)

    return train_loader

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data, transform_A=None, transform_B=None):
        # Provide a more informative assertion error message.
        assert all(data[0].shape[0] == item.shape[0] for item in data), "All data components must have the same number of samples (dimension 0)."
        self.transform_A = transform_A
        self.transform_B = transform_B

        if not isinstance(data[0], torch.Tensor):
            self.data = (torch.tensor(data[0]).float(),)
        else:
            self.data = (data[0].float(),)

        if not isinstance(data[1], torch.Tensor):
            self.data = self.data + (torch.tensor(data[1]),)
        else:
            self.data = self.data + (data[1],)

    def __getitem__(self, index):
        x = self.data[0][index]

        if self.transform_A:
            x1 = self.transform_A(x)
        else:
            x1 = x

        y = self.data[1][index]

        # No conversion here, since data is already a torch.Tensor.
        return x1, y

    def __len__(self):
        return self.data[0].shape[0]

# read dataset
def read_ECG(args):
    # Use os.path.join for better cross-platform path construction.
    da_pa = os.path.join('..', '..', 'Data', args.dataset_type + '_dataset')
    ecg_da_path = os.path.join(da_pa, 'fuer_da.npy')
    ecg_lab_path = os.path.join(da_pa, 'fuer_lab.npy')

    ecg_da = np.load(ecg_da_path)
    if os.path.isfile(ecg_lab_path):
        ecg_la = np.load(ecg_lab_path)
    else:
        ecg_la = np.zeros(ecg_da.shape[0])
        print("Warning: Label file not found. Initializing labels to zeros.")

    # whether cut
    if args.running_env == 'local':
        ecg_da = ecg_da[:200]
        ecg_la = ecg_la[:200]

    # dataset rate
    if args.dataset_type == 'finetune':
        le = int(args.labelled_ratio * ecg_da.shape[0])
        ecg_da = ecg_da[:le]
        ecg_la = ecg_la[:le]

    # whether mix leads
    if args.whe_mix_lead == 'mix':
        ecg_la = np.stack([ecg_la for _ in range(ecg_da.shape[1])], axis=-1)
        ecg_la = np.reshape(ecg_la, (-1))
        ecg_da = np.reshape(ecg_da, (-1, 1, ecg_da.shape[-1]))

    # normalization
    ecg_da = np.array([ (sample - np.mean(sample)) / np.std(sample) for sample in ecg_da ])

    return ecg_da, ecg_la
