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
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)
    
class lstm_new(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm_new, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        
    def forward(self, input, hidden=None, return_last=False):
        """
        input: a sequence of vector
        """
        embedded = []
        for i in input:
            embedded.append(self.embed(i.view(-1, self.input_size)))
        
        # input_lstm : (length, batch, hidden_dim)
        input_lstm = torch.stack(embedded, dim=0)
        
        # out : (seq_len, batch, num_directions * hidden_size)
        #     -> output from last layer for each step
        # h : (num_layers * num_directions, batch, hidden_size)
        #     -> hidden for each layer in last step
        # c : (num_layers * num_directions, batch, hidden_size)
        #     -> current state for each layer in last step
        if hidden:
            out, (h, c) = self.lstm(input_lstm, hidden)
        else:
            out, (h, c) = self.lstm(input_lstm)
        
        if return_last:
            return self.output(out[-1]), (h, c)
        else:
            out_list = []
            for i in range(out.shape[0]):
                out_list.append(self.output(out[i]))
            return out_list, (h, c)

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

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
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


class gaussian_lstm_new(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm_new, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers)
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.var_net = nn.Linear(hidden_size, output_size)
        
    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)
    
    def forward(self, input):
        """
        input: a sequence of vector
        """
        embedded = []
        for i in input:
            embedded.append(self.embed(i.view(-1, self.input_size)))
        
        # input_lstm : (length, batch, hidden_dim)
        input_lstm = torch.stack(embedded, dim=0)
        
        # out : (seq_len, batch, num_directions * hidden_size)
        #     -> output from last layer for each step
        # h : (num_layers * num_directions, batch, hidden_size)
        #     -> hidden for each layer in last step
        # c : (num_layers * num_directions, batch, hidden_size)
        #     -> current state for each layer in last step
        out, (h, c) = self.lstm(input_lstm)
        
        mu = self.mu_net(out[-1])
        var = self.var_net(out[-1])
        z = self.reparameterize(mu, var)
        return z, mu, var