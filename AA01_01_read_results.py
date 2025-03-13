import numpy as np
import support_based as spb
import os

fo_li = os.listdir('./save')
all_res = []
for fo in fo_li:
    path = './save/' + fo + '/'
    pretrain_res = spb.read_res(path, 'prettain-res')
    fineturn_nopretrain_res = spb.read_res(path, 'fineturn-nopretrain-res')
    fineturn_pretrain_full_res = spb.read_res(path, 'fineturn-pretrain_full-res')
    fineturn_pretrain_only_classifier_res = spb.read_res(path, 'fineturn-pretrain_only_classifier-res')

    fineturn_nopretrain_res = np.array([x[1] for x in fineturn_nopretrain_res[1]])
    fineturn_pretrain_full_res = np.array([x[1] for x in fineturn_pretrain_full_res[1]])
    fineturn_pretrain_only_classifier_res = np.array([x[1] for x in fineturn_pretrain_only_classifier_res[1]])

    me_fineturn_nopretrain_res = np.mean(fineturn_nopretrain_res[:, -50:], 0)
    me_fineturn_pretrain_full_res = np.mean(fineturn_pretrain_full_res[:, -50:], 0)
    me_fineturn_pretrain_only_classifier_res = np.mean(fineturn_pretrain_only_classifier_res[:, -50:], 0)

    args = pretrain_res[1]
    conf = np.array([fo.split('_')[0], args.mask_ratio, args.patch_size, args.emb_dim])
    met = np.stack([me_fineturn_nopretrain_res, me_fineturn_pretrain_full_res, me_fineturn_pretrain_only_classifier_res], 0)
    confs = np.stack([conf, conf, conf], 0)
    c_m = np.concatenate([confs, met], 1)

    all_res.append([fo, c_m])

al_tab = np.concatenate([x[1] for x in all_res], 0)