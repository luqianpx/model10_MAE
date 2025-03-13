import pickle
import random
import string
import numpy as np
import warnings
import os
import uuid
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, f1_score, recall_score
)

warnings.simplefilter('ignore')

def com_mul_str(args):
    """Generates a unique string identifier for an experiment."""
    str_li = [args.whe_mix_lead, args.epochs, args.batch_size, args.n_length]
    str_li.append(uuid.uuid4().hex[:4])

    return "_".join(map(str, str_li))

def cal_met(output, y, args):
    """Calculates classification metrics."""
    predicted = output.argmax(1)
    one_hot_y = np.eye(args.n_class)[y]

    acc = accuracy_score(y, predicted)
    precision = precision_score(y, predicted, average='macro')
    recall = recall_score(y, predicted, average='macro')
    F1 = f1_score(y, predicted, average='macro')

    try:
        auc = roc_auc_score(one_hot_y, output, average="macro", multi_class="ovr")
    except ValueError:
        auc = float(0)

    prc = average_precision_score(one_hot_y, output, average="macro")

    return np.array([acc, precision, recall, F1, auc, prc])

def save_res(sa_fo, na, res):
    """Saves results using pickle."""
    fi_na = os.path.join(sa_fo, na) 
    with open(fi_na, 'wb') as fi:
        pickle.dump(res, fi)

def read_res(sa_fo, na):
    """Reads results using pickle."""
    fi_na = os.path.join(sa_fo, na)
    with open(fi_na, 'rb') as fi:
        return pickle.load(fi)
