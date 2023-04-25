import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=2048, z_dim=512):
        super(Encoder, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.fc_mu = torch.nn.Linear(hidden_dim, z_dim)
        self.fc_sigma = torch.nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # formula (6)
        _, (h, _) = self.rnn(x) 
        mu = self.fc_mu(h)
        # formula (7)
        log_var = (self.fc_sigma(h)+1).log()
        return mu, log_var

class Decoder(torch.nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        pass

class MusicVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(MusicVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)
        pass


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, MusicVAE):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'vae.th'))
    raise ValueError("model type '%s' invalid "%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = MusicVAE()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'vae.th'), map_location='cpu'))
    return r