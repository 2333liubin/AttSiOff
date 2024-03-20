import os
os.environ['CUDA_VISIBLE_dEVICES']='0, 1, 2, 3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import warnings
warnings.filterwarnings('ignore')
import torch
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
from torch import nn
import numpy as np
import pdb
import argparse
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score

from load_data import get_dataloader_rt_or_inter
from utils import *
modules_attn = __import__('model')

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=126)
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--sample_internal', type=int, default=10)
args = parser.parse_args()


class K_Train_Test():
    def __init__(self,args):
        super(K_Train_Test, self).__init__()
        self.args = args

    def model_train(self, train_loader, test_loader, testset_type):
        print(' --- testset type : {} ------'.format(testset_type))
        model = getattr(modules_attn, 'RNAFM_SIPRED_2')(dp=0.1, device=device).to(torch.float32).to(device)
        
        crepochion_mae = torch.nn.L1Loss(reduction='mean')
        crepochion_mse = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=4)

        model_name = 'model_' + testset_type.replace('_', '') + '.pth.tar'
        save_path = os.path.join('./output/', str(testset_type))
        os.makedirs(save_path, exist_ok=True)
        early_stopping = EarlyStopping(save_path, model_save_name=model_name, patience=20, greater=True)  # 早停的指标为pcc，所以得分越大越好

        
        for epoch in range(self.args.n_epochs):
            model.train()

            for id, inputs in enumerate(train_loader):
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device).to(torch.float32)

                pred = model(inputs)
                label = inputs['inhibit']
                loss = crepochion_mse(label, pred)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()

            scheduler.step()

            model.eval()
            with torch.no_grad():
                for id, inputs in enumerate(test_loader):
                    for k in inputs.keys():
                        inputs[k] = inputs[k].to(device).to(torch.float32)

                    pred = model(inputs)
                    label = inputs['inhibit']
                    pred = np.array(pred.detach().cpu().numpy()).reshape(1, -1)[0]
                    label = np.array(label.detach().cpu().numpy()).reshape(1, -1)[0]
                    PCC, SPCC = round(stats.pearsonr(pred,label)[0], 3), round(stats.spearmanr(pred,label)[0], 3)

            early_stopping(SPCC, epoch, model)
            if early_stopping.early_stop:
                break


        model.load_state_dict(torch.load(early_stopping.best_model_path), strict=True)
        model.eval()

        with torch.no_grad():
            for id, inputs in enumerate(test_loader):
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device).to(torch.float32)

                pred = model(inputs)
                label = inputs['inhibit']
                pred = np.array(pred.detach().cpu().numpy()).reshape(1, -1)[0]
                label = np.array(label.detach().cpu().numpy()).reshape(1, -1)[0]
                PCC, SPCC = round(stats.pearsonr(pred,label)[0], 3), round(stats.spearmanr(pred,label)[0], 3)
                # MCC = matthews_corrcoef((label>0.7), (pred>0.7))
                AUC = roc_auc_score((label>0.7), pred)
                
            logging('epoch {}  ====>> pcc:{:.4f}, spcc:{:.4f}, auc:{:.4f}'.format(
                epoch, PCC, SPCC, AUC))
            print('epoch {}  ====>> pcc:{:.4f}, spcc:{:.4f}, auc:{:.4f} \n\n'.format(
                epoch, PCC, SPCC, AUC))
            # plot_scatter_curve(pred,label,save_path,title='test_pcc_{}_spcc_{}'.format(PCC, SPCC))

       
    def k_cross(self):

        for k in range(args.sample_internal):
            train_loader, test_loader = get_dataloader_rt_or_inter(self.args, str(k)+'-inter')
            self.model_train(train_loader, test_loader, str(k)+'-inter')

        for test_dataset in ['T', 'R']:
            train_loader, test_loader = get_dataloader_rt_or_inter(self.args, test_dataset)
            self.model_train(train_loader, test_loader, test_dataset)

        
def main(args):
    k_cross_course = K_Train_Test(args)
    k_cross_course.k_cross()
    log()
    print('\n task finished!! \n')

if __name__ == '__main__':
    main(args)

