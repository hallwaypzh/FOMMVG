from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        #self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(batch_size, self.hidden_size).cuda(),
                           torch.zeros(batch_size, self.hidden_size).cuda()))
        self.hidden = hidden
        return hidden

    def init_hidden_(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(batch_size, self.hidden_size).cuda(),
                           torch.zeros(batch_size, self.hidden_size).cuda()))
        self.hidden = hidden
        #return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)


class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(batch_size, self.hidden_size).cuda(),
                           torch.zeros(batch_size, self.hidden_size).cuda()))
        self.hidden = hidden
        return hidden

    def init_hidden_(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(self.batch_size, self.hidden_size).cuda(),
                           torch.zeros(self.batch_size, self.hidden_size).cuda()))
        self.hidden = hidden
        #return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        #return eps.add_(mu)
        #return eps.mul(logvar)
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
            

class gaussian_bilstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.fw_lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.bw_lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.fw_hidden = self.init_hidden()
        self.bw_hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def init_hidden_(self):
        fw_hidden = []
        bw_hidden = []
        for i in range(self.n_layers):
            fw_hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                              Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
            bw_hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                              Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))

        self.fw_hidden = fw_hidden
        self.bw_hidden = bw_hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def one_step(self, input, direction="forward"):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded

        for i in range(self.n_layers):
            if direction == "forward":
                self.fw_hidden[i] = self.fw_lstm[i](h_in, self.fw_hidden[i])
                h_in = self.fw_hidden[i][0]
            else:
                self.bw_hidden[i] = self.bw_lstm[i](h_in, self.bw_hidden[i])
                h_in = self.bw_hidden[i][0]

        return h_in

    def forward(self, input):
        fw_h_in = self.one_step(input, "forward")
        bw_h_in = self.one_step(input, "forward")

        h_in = torch.cat([fw_h_in, bw_h_in], 1)

        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers, act=nn.LeakyReLU(negative_slope=0.2), last_act=False):
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


class MyLSTM(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed_h = nn.Linear(output_size, hidden_size)
        self.embed_c = nn.Linear(output_size, hidden_size)
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Tanh())
    
    def forward(self, x, noises):
        h0 = self.embed_h(x)
        c0 = self.embed_c(x)
        #c0 = torch.zeros_like(h0, device=x.device)
        h, c = h0, c0
        dxs = []
        for epsilon in noises:
            (h, c) = self.lstm(epsilon, (h, c))
            dx = self.output(h)
            dxs.append(dx)
        return dxs 

class ConvDiscriminator(nn.Module):
    
    def __init__(self, x_dim, t_dim, kernel_size, act=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        self.act = act
        self.kernel_size = kernel_size
        self.x_dim = x_dim 
        self.t_dim = t_dim 
        
        layers = [nn.Conv1d(in_channels=x_dim, out_channels=100, kernel_size=kernel_size)]
        #import pdb; pdb.set_trace()
        layers.append(self.act)
        layers.append(nn.Conv1d(in_channels=100, out_channels=50, kernel_size=kernel_size))
        layers.append(self.act)
        layers.append(nn.Conv1d(in_channels=50, out_channels=10, kernel_size=kernel_size))
        layers.append(self.act)  
        self.convs = nn.Sequential(*layers)
        
        layers = [nn.Linear(30, 30)]
        layers.append(self.act)
        layers.append(nn.Linear(30, 1))
        self.linears = nn.Sequential(*layers)
        #self.linears = nn.Linear(40, 40)
        
        # layers = [nn.Conv1d(in_channels=x_dim, out_channels=x_dim, kernel_size=kernel_size)]
        # layers.append(self.act)
        # layers.append(nn.Conv1d(in_channels=x_dim, out_channels=50, kernel_size=kernel_size))
        # layers.append(self.act)
        # self.convs = nn.Sequential(*layers)
        # layers = [nn.Linear(x_dim, 50)]
        # layers.append(self.act)
        # layers.append(nn.Linear(50, 1))
        # self.linears = nn.Sequential(*layers)
    
    def forward(self, x):
        assert len(x.shape) == 3
        
        assert x.shape[1] == self.x_dim 
        assert x.shape[2] == self.t_dim
        
        # out = self.convs(x)
        # #import pdb; pdb.set_trace()
        # return self.linear(out)
        out = self.convs(x)
        b, _, _ = out.shape

        return self.linears(out.view(b, -1))
        

class MyLSTMx(nn.Module):
    
    def __init__(self, input_size, output_size, x_size, hidden_size, n_layers, require_bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed_h = nn.Linear(x_size, hidden_size)
        self.embed_c = nn.Linear(x_size, hidden_size)
        self.lstm = nn.LSTMCell(input_size, hidden_size, bias=require_bias)
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Tanh())
    
    def forward(self, x, noises):
        h0 = self.embed_h(x)
        c0 = self.embed_c(x)
        #c0 = torch.zeros_like(h0, device=x.device)
        h, c = h0, c0
        dxs = []
        for epsilon in noises:
            (h, c) = self.lstm(epsilon, (h, c))
            dx = self.output(h)
            dxs.append(dx)
        return dxs 


class LSTMDiscriminator(nn.Module):
    
    def __init__(self, x_dim, h_dim, mlp_layers, act=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        self.x_dim = x_dim 
        self.h_dim = h_dim 
        self.mlp_layers = mlp_layers
        self.act = act 
        self.lstm = nn.LSTMCell(x_dim, h_dim)
        layers = []
        for i in range(mlp_layers-1):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(act)
        layers.append(nn.Linear(h_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: LxBxN
        L, B, _  = x.shape
        h, c = torch.zeros(B, self.h_dim, device=x.device), torch.zeros(B, self.h_dim, device=x.device)
        for i in range(L):
            h, c = self.lstm(x[i], (h, c))
        return self.mlp(h)

