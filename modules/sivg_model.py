from os import stat
import torch 
import torch.nn as nn 
from torch import fake_quantize_per_channel_affine, optim 

import modules.p2plstm as lstm_models
import utils
import random

class Direction(nn.Module):
    def __init__(self, m, n):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(m, n))

    def forward(self, input):
        # input: (bs*t) x 512
        weight = self.weight + 1e-8 
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out


import numpy as np 

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers, act=nn.LeakyReLU(negative_slope=0.2), last_act=False, dropout=False):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.n_layers = n_layers         
        
        layers = [nn.Linear(input_size, hidden_size)]
        for i in range(n_layers):
            layers.append(act)
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(act)
        layers.append(nn.Linear(hidden_size, output_size))
        if last_act:
            layers.append(act)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
class KLCriterion(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        """KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2))"""
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum()

class KPSeqGAN_MLP_PCAMD(nn.Module):
    # use x0 as h_0, c0 for forward LSTM!
    # similar to MOCOGAN-HD
    def __init__(self, x_dim=100, n_dim=32, h_dim=32,
                 pca_paths='pca_states/RAV_KPS_PCA_30.npy',
                 require_bias=False, opt=None):

        super().__init__()
        self.x_dim                = x_dim 
        self.n_dim                = n_dim
        self.h_dim                = h_dim
        self.opt                  = opt
        self.D_len                = 1
        
        pca_states                = np.load(pca_paths, allow_pickle=True).item()
        self.pca_comp             = torch.from_numpy(pca_states['comp']).float().cuda()
        self.pca_stdev            = torch.from_numpy(pca_states['stdev']).float().cuda()
        self.n_comp               = self.pca_comp.shape[0]
        # subnetworks
        self.frnn = nn.LSTMCell(n_dim, h_dim)
        self.ench = lstm_models.MLP(x_dim, h_dim, h_dim, 1)
        self.encc = lstm_models.MLP(x_dim, h_dim, h_dim, 1)
        self.pred = lstm_models.MLP(h_dim, self.n_comp, self.n_comp, 0)
        self.D = MLP(x_dim, 50, 1, 1)
        
        # optimizer
        self.frnn_optimizer = optim.Adam(self.frnn.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))
        self.pred_optimizer = optim.Adam(self.pred.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))
        self.D_optimizer    = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.ench_optimizer = optim.Adam(self.ench.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))
        self.encc_optimizer = optim.Adam(self.encc.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))
        # criterions
        self.mse_criterion = nn.MSELoss() # recon and cpc
        self.kl_criterion  = KLCriterion()
        
    def p2p_generate(self, x, len_output):
        """Point-to-Point Generation given input sequence. Generate *1* sample for each input sequence.
        params:
            x: input sequence
            len_output: length of the generated sequence
            eval_cp_ix: cp_ix of the output sequence. usually it is len_output-1
            model_mode:
                - full:      post then prior
                - posterior: all use posterior
                - prior:     all use prior
        """
        with torch.no_grad():
            pred_xs, pred_params_mu, pred_params_sigma = self.G_step(x[0], len_output-1)
            return [x for x in pred_xs]

    def G_step(self, x0, seq_len):
        B, N = x0.shape
        noises = torch.randn(seq_len, B, self.n_dim, device=x0.device)
        fh0, fc0 = self.ench(x0), self.encc(x0)
        # collect forword hidden states
        fh, fc = fh0, fc0
        fhs = []
        fcs = []
        for noise in noises:
            fh, fc = self.frnn(noise, (fh, fc))
            fhs.append(fh)
            fcs.append(fc)
        fhs = torch.stack(fhs)
        fcs = torch.stack(fcs)
        pred_params = self.pred(fhs)
        pred_params_mu = pred_params.view(-1, self.n_comp).mean(dim=0)
        pred_params_sigma = pred_params.view(-1, self.n_comp).var(dim=0, unbiased=True)

        pred_delta_xs = torch.matmul(torch.diag_embed(pred_params.view(-1, self.n_comp)), self.pca_comp).sum(dim=1).view(-1, B, N)
        pred_xs = [x0]
        for delta_x in pred_delta_xs:
            pred_xs.append((pred_xs[-1] + delta_x).clamp(-1, 1.))
        
        pred_xs = torch.stack(pred_xs)
        
        return pred_xs, pred_params_mu, pred_params_sigma

    def forward_D(self, data_real, data_fake, for_D):
        loss_real = 0.   
        seq_len = data_fake.shape[0]
        
        if not (data_real is None):
            real_inp = (data_real[1:] - data_real[:-1]).detach()
            logit_real = self.D(real_inp)
            #loss_real = torch.nn.functional.softplus(-logit_real)
            loss_real  = (logit_real - 1.) ** 2

        if for_D:
            loss_fake = 0.
            fake_inp = (data_fake[1:self.D_len+1] - data_fake[:self.D_len]).view(-1, self.x_dim)
            logit_fake = self.D(fake_inp)
            loss_fake  = (logit_fake - 0.) ** 2
        else:
            fake_inp = (data_fake[1:self.D_len+1] - data_fake[:self.D_len]).view(-1, self.x_dim)
            logit_fake = self.D(fake_inp)
            #loss_fake = torch.nn.functional.softplus(-logit_fake)
            loss_fake  = (logit_fake - 1.) ** 2
        #loss_reg += -(logit_fake - 0.5) ** 2
            
        return loss_real, loss_fake
    
    def forward(self, real_x, x0, G_len=16):
        # x: TxBxN
        # update G 
        self.zero_grad_G()
        fake_x, pred_params_mu, pred_params_sigma = self.G_step(x0, G_len)
        _, loss_fake_G = self.forward_D(None, fake_x, for_D=False)
        
        loss_G = torch.mean(loss_fake_G)

        dist_loss = self.kl_criterion(pred_params_mu, pred_params_sigma, \
                                      torch.zeros_like(pred_params_mu, device=pred_params_mu.device), 
                                      self.pca_stdev)
        loss = loss_G + self.opt.weight_lm * dist_loss
        loss.backward()
        self.update_G()
        
        # update D
        self.D_optimizer.zero_grad()
        fake_x, _, _ = self.G_step(x0, G_len)
        #fake_x = self.forward_G(x0, seq_len)
        loss_real_D, loss_fake_D = self.forward_D(real_x, fake_x.detach(), for_D=True)
        
        #import pdb; pdb.set_trace()
        loss_D = 0.5 * (loss_real_D.mean() + loss_fake_D.mean()) 

        loss_D.backward()
        self.D_optimizer.step() 

        loss_dict = dict()
        loss_dict['loss_G'] = loss_fake_G.mean().data.cpu().numpy()
        loss_dict['loss_D_real'] = loss_real_D.mean().data.cpu().numpy()
        loss_dict['loss_D_fake'] = loss_fake_D.mean().data.cpu().numpy()
        loss_dict['loss_vae']   =  dist_loss.data.cpu().numpy()
        loss_dict['logit_real'] = (1 - loss_real_D.mean() ** 0.5).data.cpu().numpy()
        loss_dict['logit_fake'] = (loss_fake_D.mean() ** 0.5).data.cpu().numpy()
        # print(loss_G.item(), loss_D.item())
        # import pdb; pdb.set_trace()
        return loss_dict 
    
    def update_G(self):
        self.frnn_optimizer.step()
        self.pred_optimizer.step()
        self.ench_optimizer.step()
        self.encc_optimizer.step()
        
    def zero_grad_G(self):
        self.frnn.zero_grad()
        self.pred.zero_grad()
        self.ench.zero_grad()
        self.encc.zero_grad()

    def save(self, fname, epoch):
        # cannot torch.save with module
        states = {
            'frnn': self.frnn.state_dict(),
            'pred': self.pred.state_dict(),
            'D': self.D.state_dict(),
            'ench': self.ench.state_dict(),
            'encc': self.encc.state_dict(),
            'frnn_opt': self.frnn_optimizer.state_dict(),
            'pred_opt': self.pred_optimizer.state_dict(),
            'D_opt': self.D_optimizer.state_dict(),
            'ench_opt': self.ench_optimizer.state_dict(),
            'encc_opt': self.encc_optimizer.state_dict(),
            'epoch': epoch,
            'opt': self.opt,
        }
        torch.save(states, fname)


    def load(self, pth=None, states=None, for_testing=False):
        # """ load from pth or states directly """
        if states is None:
            states = torch.load(pth)
        self.frnn.load_state_dict(states['frnn'])
        self.pred.load_state_dict(states['pred'])
        self.D.load_state_dict(states['D'])
        self.encc.load_state_dict(states['ench'])
        self.ench.load_state_dict(states['encc'])
        if not for_testing:
            self.frnn_optimizer.load_state_dict(states['frnn_opt'])
            self.pred_optimizer.load_state_dict(states['pred_opt'])
            self.D_optimizer.load_state_dict(states['D_opt'])
            self.encc_optimizer.load_state_dict(states['encc_opt'])
            self.ench_optimizer.load_state_dict(states['ench_opt'])
            self.opt = states['opt']
        start_epoch = states['epoch'] + 1
        return start_epoch
