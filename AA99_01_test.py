import numpy as np
import support_based as spb

path = 'E:/Project20_ECG_foundation_model/Model/model10_MAE/save/mix_30_5_2500_xC4l/'
pretrain_res = spb.read_res(path, 'prettain-res')
fineturn_nopretrain_res = spb.read_res(path, 'fineturn-nopretrain-res')
fineturn_pretrain_full_res = spb.read_res(path, 'fineturn-pretrain_full-res')
fineturn_pretrain_only_classifier_res = spb.read_res(path, 'fineturn-pretrain_only_classifier-res')

fineturn_nopretrain_res = np.array([x[1] for x in fineturn_nopretrain_res[1]])
fineturn_pretrain_full_res = np.array([x[1] for x in fineturn_pretrain_full_res[1]])
fineturn_pretrain_only_classifier_res = np.array([x[1] for x in fineturn_pretrain_only_classifier_res[1]])