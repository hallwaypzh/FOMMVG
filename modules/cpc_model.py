import torch 
import torch.nn as nn 
from torch import optim 

import modules.p2plstm as lstm_models
import utils

class P2PModelBDG_MLP_LM(nn.Module):
    # use x0 as h_0, c0 for forward LSTM!
    # use x1 as h_0, c0 for backward LSTM!
    # similar to MOCOGAN-HD
    def __init__(self, batch_size=100, x_dim=100, n_dim=50, rnn_size=100, 
                 rnn_layers=2, training_seq_len=16, opt=None):

        super().__init__()
        self.batch_size           = batch_size
        self.x_dim                = x_dim 
        self.n_dim                = n_dim
        self.rnn_size             = rnn_size 
        self.rnn_layers           = rnn_layers
        self.opt                  = opt
        self.training_seq_len     = training_seq_len
        self.D_len                = 1

        # subnetworks
        self.E  = lstm_models.MLP(x_dim, 2*rnn_size, 2*rnn_size, 1)
        self.frnn = nn.LSTMCell(n_dim, rnn_size)
        self.brnn = nn.LSTMCell(n_dim, rnn_size)
        self.pred = lstm_models.MLP(2*rnn_size, x_dim, x_dim, 1)
        self.H = lstm_models.MLP(x_dim, n_dim, n_dim, 1)
        self.D = lstm_models.MLP(4*x_dim, 100, 1, 1)
        
        # optimizer
        self.frnn_optimizer = optim.Adam(self.frnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.brnn_optimizer = optim.Adam(self.brnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.pred_optimizer = optim.Adam(self.pred.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.E_optimizer    = optim.Adam(self.E.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.D_optimizer    = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.H_optimizer    = optim.Adam(self.H.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))        
        self.init_weight()
        
        self.mse_criterion = nn.MSELoss()

    def init_weight(self):
        self.frnn.apply(utils.init_weights)
        self.brnn.apply(utils.init_weights)
        self.pred.apply(utils.init_weights)
        
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
            pred_xs, _, _ = self.G_step(x[0], x[-1], x.shape[0]-2)
            return [x for x in pred_xs]
    
    def G_step(self, x0, x1, seq_len):
        B, N = x0.shape
        noises = torch.randn(seq_len, B, self.n_dim, device=x0.device)
        
        fhc = self.E(x0)
        bhc = self.E(x1)
        fh, fc = fhc[:, :self.rnn_size], fhc[:, self.rnn_size:]
        bh, bc = bhc[:, :self.rnn_size], bhc[:, self.rnn_size:]

        fhs = []
        for noise in noises:
            fh, fc = self.frnn(noise, (fh, fc))
            fhs.append(fh)
        fhs = torch.stack(fhs)

        bhs = []
        for noise in noises.flip(dims=(0,)):
            bh, bc = self.brnn(noise, (bh, bc))
            bhs.append(bh)
        bhs = torch.stack(bhs).flip(dims=(0,))

        pred_xs = self.pred(torch.cat([fhs, bhs], dim=-1))
        noise_enc = self.H(pred_xs)
        pred_xs = torch.cat([x0.unsqueeze(0), pred_xs, x1.unsqueeze(0)])
        return pred_xs, noise_enc, noises

    def forward_D(self, data_real, data_fake, for_D):
        loss_real = 0.   
        seq_len = data_fake.shape[0]
        if not (data_real is None):
            for i in range(self.D_len):
                real_inp_i = torch.cat([data_real[i], data_real[seq_len-1-i], data_real[i+1]-data_real[i], data_real[seq_len-1-i]-data_real[seq_len-2-i]], dim=-1)
                logit_real_i = self.D(real_inp_i)
                loss_real += torch.nn.functional.softplus(-logit_real_i)
            loss_real /= self.D_len    
        if for_D:
            loss_fake = 0.
            for i in range(self.D_len):
                fake_inp_i = torch.cat([data_fake[i], data_fake[seq_len-1-i], data_fake[i+1] - data_fake[i], data_fake[seq_len-1-i] - data_fake[seq_len-2-i]], dim=-1)
                logit_fake_i = self.D(fake_inp_i)
                loss_fake += torch.nn.functional.softplus(logit_fake_i)
            loss_fake /= self.D_len 
        else:
            loss_fake = 0.
            for i in range(self.D_len):
                fake_inp_i = torch.cat([data_fake[i].detach(), data_fake[seq_len-1-i].detach(), data_fake[i+1] - data_fake[i], data_fake[seq_len-1-i] - data_fake[seq_len-2-i]], dim=-1)
                logit_fake_i = self.D(fake_inp_i)
                loss_fake += torch.nn.functional.softplus(-logit_fake_i)
            loss_fake /= self.D_len 
        return loss_real, loss_fake 
    
    def forward(self, x0, x1, real_x, iter=1):
        # x: TxBxN
        seq_len = real_x.shape[0] - 2
        # update G 
        self.zero_grad_G()
        fake_x, noises_enc, noises = self.G_step(x0, x1, seq_len)
        _, loss_fake_G = self.forward_D(real_x, fake_x, for_D=False)        
        loss_G = loss_fake_G.mean()
        loss_motion = -torch.nn.functional.cosine_similarity(
                        torch.stack([noises_enc[:self.D_len], noises_enc[-self.D_len:]]),
                        torch.stack([noises[:self.D_len], noises[-self.D_len:]]), dim=-1).mean()
        loss_recons = (real_x - fake_x).abs().mean()
        loss = loss_G + loss_motion * self.opt.weight_lm  + loss_recons * self.opt.weight_recons
        loss.backward()
        self.update_G()
        # update D
        self.D_optimizer.zero_grad()
        loss_real_D, loss_fake_D = self.forward_D(real_x, fake_x.detach(), for_D=True)
        loss_D = 0.5 * (loss_real_D + loss_fake_D).mean()
        loss_D.backward()
        self.D_optimizer.step() 
        
        loss_dict = dict()
        loss_dict['loss_G'] = loss_G.data.cpu().numpy()
        loss_dict['loss_D_real'] = loss_real_D.mean().data.cpu().numpy()
        loss_dict['loss_D_fake'] = loss_fake_D.mean().data.cpu().numpy()
        loss_dict['loss_motion'] = loss_motion.data.cpu().numpy()
        loss_dict['loss_recon'] = 0.
        return loss_dict 
    
    def update_G(self):
        self.frnn_optimizer.step()
        self.brnn_optimizer.step()
        self.pred_optimizer.step()
        self.H_optimizer.step()
        self.E_optimizer.step()
    
    def zero_grad_G(self):
        self.frnn.zero_grad()
        self.brnn.zero_grad()
        self.pred.zero_grad() 
        self.H.zero_grad()
        self.E.zero_grad()

    def save(self, fname, epoch):
        # cannot torch.save with module
        states = {
            'frnn': self.frnn.state_dict(),
            'brnn': self.brnn.state_dict(),
            'pred': self.pred.state_dict(),
            'D': self.D.state_dict(),
            'H': self.H.state_dict(),
            'E': self.E.state_dict(),
            'frnn_opt': self.frnn_optimizer.state_dict(),
            'brnn_opt': self.brnn_optimizer.state_dict(),
            'pred_opt': self.pred_optimizer.state_dict(),
            'D_opt': self.D_optimizer.state_dict(),
            'H_opt': self.H_optimizer.state_dict(),
            'E_opt': self.E_optimizer.state_dict(),
            'epoch': epoch,
            'opt': self.opt,
            }
        torch.save(states, fname)


    def load(self, pth=None, states=None):
        # """ load from pth or states directly """
        if states is None:
            states = torch.load(pth)
        self.frnn.load_state_dict(states['frnn'])
        self.brnn.load_state_dict(states['brnn'])
        self.pred.load_state_dict(states['pred'])
        self.D.load_state_dict(states['D'])
        self.H.load_state_dict(states['H'])
        self.E.load_state_dict(states['E'])
        
        self.frnn_optimizer.load_state_dict(states['frnn_opt'])
        self.brnn_optimizer.load_state_dict(states['brnn_opt'])
        self.pred_optimizer.load_state_dict(states['pred_opt'])
        self.D_optimizer.load_state_dict(states['D_opt'])
        self.H_optimizer.load_state_dict(states['H_opt'])
        self.E_optimizer.load_state_dict(states['E_opt'])
        
        self.opt = states['opt']
        start_epoch = states['epoch'] + 1
        return start_epoch
