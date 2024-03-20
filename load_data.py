import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import product
import math
import RNA
from utils import readFaRNAFOLD
import sklearn
import subprocess
import os

########################## 各种特征 #############################


def get_gc_sterch(seq):  # 1,
    max_len, tem_len = 0, 0
    for i in range(len(seq)):
        if seq[i] == 'G' or seq[i] == 'C':
            tem_len += 1
            max_len = max(max_len, tem_len)
        else:
            tem_len = 0

    result = round((max_len / len(seq)), 3)
    return np.array([result])[:, np.newaxis]

def get_gc_percentage(seq):  # 1,
    result = round(((seq.count('C') + seq.count('G')) / len(seq)), 3)
    return np.array([result])[:, np.newaxis]

def get_single_comp_percent(seq):  # 4,
    nt_percent = []
    for base_i in list(['A', 'G', 'C', 'U']):
        nt_percent.append(round((seq.count(base_i) / len(seq)), 3))
    return np.array(nt_percent)[:, np.newaxis]

def get_di_comp_percent(seq):  # 16,
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=2))
    di_nt_percent = []
    for pmt_i in pmt:
        di_nt = pmt_i[0] + pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt) / (len(seq) - 1)), 3))
    return np.array(di_nt_percent)[:, np.newaxis]

def get_tri_comp_percent(seq):  # 64,
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=3))
    tri_nt_percent = []
    for pmt_i in pmt:
        tri_nt = pmt_i[0] + pmt_i[1] + pmt_i[2]
        tri_nt_percent.append(round((seq.count(tri_nt) / (len(seq) - 2)), 3))
    return np.array(tri_nt_percent)[:, np.newaxis]


def secondary_struct(seq):  # 2+1
   
    def _percentage(if_paired):
        paired_percent = (if_paired.count('(') + if_paired.count(')')) / len(if_paired)
        unpaired_percent = (if_paired.count('.')) / len(if_paired)
        return np.array([[paired_percent], [unpaired_percent]])

    paired_seq, min_free_energy = RNA.fold(seq)
    return _percentage(paired_seq), np.array([min_free_energy])[:, np.newaxis]


def score_seq_by_pssm(pssm, seq):  # 1,
    nt_order = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    ind_all = list(range(0, len(seq)))
    scores = [pssm[nt_order[nt], i] for nt, i in zip(seq, ind_all)]
    log_score = sum([-math.log2(i) for i in scores])
    return np.array([log_score])[:, np.newaxis]

def gibbs_energy(seq):  # 20 
    energy_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'table': np.array(
        [[-0.93, -2.24, -2.08, -1.1],
         [-2.11, -3.26, -2.36, -2.08],
         [-2.35, -3.42, -3.26, -2.24],
         [-1.33, -2.35, -2.11, -0.93]])}

    result = []
    for i in range(len(seq)-1):
        index_1 = energy_dict.get(seq[i])
        index_2 = energy_dict.get(seq[i + 1])
        result.append(energy_dict['table'][index_1, index_2])

    result.append(np.array(result).sum().round(3))
    result.append((result[0] - result[-2]).round(3)) 

    result = np.array(result)[:, np.newaxis]
    return result  # / abs(result).max()


def create_pssm(train_seq):
    # train_seq = [seq.split('!') for seq in train_seq]  #通过split方式将字符串整体转为list，如果用list的话会分割字符串为单个字符
    train_seq = [list(seq.upper()) for seq in train_seq]
    train_seq = np.array(train_seq)

    nr, nc = np.shape(train_seq)
    pseudocount = nr ** 0.5  # Introduce a pseudocount (sqrt(N)) to make sure that we do not end up with a score of 0
    bases = ['A', 'G', 'C', 'U']
    pssm = []
    for c in range(0, nc):
        col_c = train_seq[:, c].tolist()
        f_A = round(((col_c.count('A') + pseudocount) / (nr + pseudocount)), 3)
        f_G = round(((col_c.count('G') + pseudocount) / (nr + pseudocount)), 3)
        f_C = round(((col_c.count('C') + pseudocount) / (nr + pseudocount)), 3)
        f_U = round(((col_c.count('U') + pseudocount) / (nr + pseudocount)), 3)
        pssm.append([f_A, f_G, f_C, f_U])
    pssm = np.array(pssm)
    pssm = pssm.transpose()
    return pssm


def process_mrna_RNAFM(ori_mrna, rnafm_mrna):
    # replace the encoding vector of '.' at the ends of mrna with [0.05...0.05]
    front_dot_num, back_dot_num = ori_mrna[:20].count('.'), ori_mrna[-20:].count('.')
    if front_dot_num != 0:
        rnafm_mrna = np.concatenate([np.ones((front_dot_num, 640)) * 0.05, rnafm_mrna], axis=0)
    if back_dot_num != 0:
        rnafm_mrna = np.concatenate([rnafm_mrna, np.ones((back_dot_num, 640)) * 0.05], axis=0)
    # pdb.set_trace()
    assert rnafm_mrna.shape[0] == 59
    return rnafm_mrna

#### 子函数结束----------------------------------------------------------------


def load_sirna(args, dataset, pssm):

    result_dict = []
    ori_as, mrna, s_biopredsi, dsir, iscore, inhibition, idx = \
        dataset['seq'], dataset['mrna'], dataset['s-biopredsi'], dataset['dsir'], \
            dataset['i-score'], dataset['inhibition'], dataset['RNAFM_ind']

    ori_as = np.array([ori_as[i].upper() for i in range(ori_as.shape[0])])

    for i in range(ori_as.shape[0]):
        dict_tem = {}

        dict_tem['rnafm_encode'] = np.load('./data/RNAFM_sirna/'+str(idx[i]).zfill(4)+'.npy', allow_pickle=True)  # 21, 640
        tmp = np.load('./data/RNAFM_mrna/' + str(idx[i]).zfill(4) + '.npy',allow_pickle=True)
        dict_tem['rnafm_encode_mrna'] = process_mrna_RNAFM(mrna[i], tmp) # 59, 640

        dict_tem['sirna_gibbs_energy'] = gibbs_energy(ori_as[i][:19])  # 20,1
        dict_tem['pssm_score'] = score_seq_by_pssm(pssm, ori_as[i])  # 1,
        dict_tem['sirna_second_percent'], dict_tem['sirna_second_energy'] = secondary_struct(ori_as[i])  # 2,1
        dict_tem['tri_nt_percent'] = get_tri_comp_percent(ori_as[i])  # 64,
        dict_tem['di_nt_percent'] = get_di_comp_percent(ori_as[i])  # 16,
        dict_tem['single_nt_percent'] = get_single_comp_percent(ori_as[i])  # 4,
        dict_tem['gc_content'] = get_gc_percentage(ori_as[i])  # 1,
        dict_tem['gc_sterch'] = get_gc_sterch(ori_as[i])  # 1,

        dict_tem['s-biopredsi'], dict_tem['dsir'], dict_tem['i-score'] = np.array([s_biopredsi[i]]), \
                                                                         np.array([dsir[i]]), \
                                                                         np.array([iscore[i]])

        dict_tem['inhibit'] = inhibition[i]

        result_dict.append(dict_tem)

    return result_dict

class Generate_dataset(Dataset):
    def __init__(self, args, dataset, pssm):
        super().__init__()
        self.dict = load_sirna(args, dataset, pssm)
    def __getitem__(self, index):
        return self.dict[index]
    def __len__(self):
        return len(self.dict)


def get_dataloader_for_all_condition(args, all_data, train_indx, test_indx):
    def getdict(data):
        return {'seq': np.array(data['Antisense']),
                'mrna': np.array(data['mrna']),
                's-biopredsi': np.array(data['s-Biopredsi']),
                'dsir': np.array(data['DSIR'])/100.,
                'i-score': np.array(data['i-score'])/100.,
                'inhibition': np.array(data['inhibition']),
                'RNAFM_ind': np.array(data['RNAFM_ind']),
                }

    test_data = pd.DataFrame([all_data.iloc[i, :] for i in test_indx])
    train_data = pd.DataFrame([all_data.iloc[i, :] for i in train_indx])
    result_train, result_test = getdict(train_data), getdict(test_data)
    pssm_as_train, pssm_as_test = create_pssm(result_train['seq']), create_pssm(result_test['seq'])

    trainset = Generate_dataset(args, result_train, pssm_as_train)
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    testset = Generate_dataset(args, result_test, pssm_as_test)
    test_loader = DataLoader(testset, batch_size=testset.__len__(), shuffle=False, drop_last=False)

    return train_loader, test_loader

def get_dataloader_rt_or_inter(args, test_type, datafile = "./data/normalized_sirna_with_mrna.csv"):
    all_data = pd.read_csv(datafile, sep=',', header=0)
    all_sources = np.array(all_data['source_paper'])
    rows = all_data.shape[0]
    train_indx, test_indx = None, None

    if test_type.endswith('inter'):
        start_ind = int(test_type.split('-')[0])
        test_indx = np.arange(start_ind, rows, args.sample_internal)
    elif test_type == 'R' or test_type == 'T':
        test_indx = np.array(np.where(all_sources == test_type)[0])
    else:
        raise ValueError('unsupported test type : {}'.format(test_type))

    train_indx = np.array(list(set(list(range(0, rows))) - set(list(test_indx))))
    train_loader, test_loader = get_dataloader_for_all_condition(args, all_data, train_indx, test_indx)
    return train_loader, test_loader
