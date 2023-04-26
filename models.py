import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=2048, z_dim=512):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc_mu = torch.nn.Linear(hidden_dim*2, z_dim)
        self.fc_sigma = torch.nn.Linear(hidden_dim*2, z_dim)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        #양방향 h vector stack
        h = torch.cat((h[-2], h[-1]), dim=1)
        
        #mu 와 sigma(log_var) 처리하여 latent space의 특성 반환
        mu = self.fc_mu(h)
        log_var = (self.fc_sigma(h)+1).log()
        return mu, log_var


class HierarchicalDecoder(torch.nn.Module):
    def __init__(self, z_dim=512,
                 hidden_dim = 1024,
                 condoctor_output_dim=512,
                 output_dim=32,
                 num_layers=2,
                 U=16):
        super().__init__()
        self.U  = U
        self.fc0 = torch.nn.Sequential(*[
            torch.nn.Linear(z_dim//U, condoctor_output_dim),
            torch.nn.Tanh()
        ])
        self.fc1 = torch.nn.Sequential(*[
            torch.nn.Linear(condoctor_output_dim, hidden_dim),
            torch.nn.Tanh()
        ])

        self.conductor_lstm = torch.nn.LSTM(condoctor_output_dim, condoctor_output_dim, num_layers=num_layers, batch_first=True)
        self.decoder_lstm = torch.nn.LSTM(condoctor_output_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = torch.nn.Sequential(*[
            torch.nn.Softmax(0),
            torch.nn.Linear(hidden_dim, output_dim*4//U),
        ])
        self.activation = torch.nn.Softmax(1)

    def forward(self, z):
        batch_size = z.shape[0]
        z = z.view(batch_size, self.U, -1)
        outputs = []

        # U개의 세그먼트로 분해 후 각각 RNN 적용
        for i in range(self.U):
          
          #conductor RNN
          subsequence = z[:, i, :]
          subsequence = self.fc0(subsequence)
          subsequence, (h0, c0) = self.conductor_lstm(subsequence)
          
          #initial state for decoder
          h0 = self.fc1(h0)
          c0 = torch.zeros_like(h0)
          
          #decoder RNN
          subseq_out, _ = self.decoder_lstm(subsequence, (h0, c0))
          subseq_out = self.output_layer(subseq_out)
          outputs.append(subseq_out)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs.reshape(outputs.shape[0], -1, 4)

class MusicVAE(torch.nn.Module):
    def __init__(self):
        super(MusicVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = HierarchicalDecoder()

    def forward(self, x):
      mu, log_var = self.encoder(x)

      #N(0, I)의 분포를 가정하는 eps(noise)에 sigma를 곱하여 logit 생성 후 decode
      z = mu + torch.randn_like(log_var) * log_var
      return self.decoder(z), mu, log_var
    

def save_model(model):
    from torch import save
    from os import path
    fpath = '/content/'
    if isinstance(model, MusicVAE):
        return save(model.state_dict(), path.join(fpath), 'vae.th'))
    raise ValueError("model type '%s' invalid "%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = MusicVAE()
    try:
        r.load_state_dict(load(path.join(fpath, 'vae.th'), map_location='cpu'))
    except:
        return r
    return r