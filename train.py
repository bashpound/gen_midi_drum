from models import save_model, load_model
from utils import load_data
import torch
import torch.nn.functional as F

import torch.utils.tensorboard as tb
from models import MusicVAE
from torch import optim
import numpy as np

#hyperparameter decay params

#optimizer & schedular
learning_rate = 1e-4
lr_decay_rate = 0.9999

#decay params for Beta
initial_kl_weight = 0.01
max_kl_weight = 0.2
kl_annealing_rate = 0.99999

def train(args):
    from os import path
    dataset_path = '/content/groove'
    try:
        model = load_model()
    except:
        model = MusicVAE()

    model.train()

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Adam에서 reconstruction loss의 descent가 제대로 이루어지지 않아, descent algorithm을 택하여 reconstruction loss의 descent를 강제
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    #scheduling
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)

    # split='train'인 데이터를 선별
    train_data = load_data(dataset_path, opt='train')
    gs = 0
    
    # 논문에서 지정한 loss function
    reconstruction_loss = torch.nn.CrossEntropyLoss()
    sm = F.softmax

     
    for epoch in range(1):
        for batch, data in enumerate(train_data):
            optimizer.zero_grad()
            out, mu, log_var= model(data)

            z0 = sm(mu + torch.randn_like(log_var) * log_var, 0)
            
            out, mu, log_var= model(data)
            z1 = sm(mu + torch.randn_like(log_var) * log_var, 0)

            kl_loss = F.kl_div(z0.log(), z1, reduction='batchmean')

            #loss annealed with Beta
            recon_loss = reconstruction_loss(sm(out, 1), sm(data, 1))
            
            beta = min(max_kl_weight, initial_kl_weight * np.exp(epoch * kl_annealing_rate))
            
            loss = recon_loss + beta * kl_loss
            print('kl_loss', kl_loss.item(), 'recon_loss', recon_loss.item(), 'beta', beta, 'loss', loss.item())
            #logging
            train_logger.add_scalar('train/kl_loss', kl_loss, global_step=gs)
            train_logger.add_scalar('train/recon_loss', recon_loss, global_step=gs)
            train_logger.add_scalar('train/loss', loss, global_step=gs)
            gs += 1

            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()        
    save_model(model)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-e', '--epochs', default=100)
    args = parser.parse_args()
    train(args)

