from models import CNNClassifier, save_model, ClassificationLoss, load_model
from utils import load_data
import torch
import torch.utils.tensorboard as tb
from models import MusicVAE, vae_loss
from torch import optim

input_dim = 128
hidden_dim = 256
z_dim = 64

def accuracy(outputs, data):
    pass

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = torch.nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= x.size(0) * x.size(1)
    return recon_loss + beta * kl_loss


def train(args):
    from os import path
    dataset_path = './data/groove'
    model = MusicVAE(input_dim, hidden_dim, z_dim)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    learning_rate = 1e-3
    lr_decay_rate = 0.9999
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)

    train_data, test_data = load_data(dataset_path, 'train'), load_data(dataset_path, 'test')
    device = torch.device('cpu')
    gs = 0
    
    loss = vae_loss
    ground_truth_rate = 0.2
     
    for epoch in range(int(args.epochs)):
        for batch, (data, label) in enumerate(train_data):

            outputs = model(data)
            l = loss(outputs)
            optimizer.zero_grad()
            train_logger.add_scalar('train/loss', l, global_step=gs)
            train_logger.add_scalar('train/accuracy', accuracy(outputs, data), global_step=gs)
            gs += 1
            l.backward()
            optimizer.step()
            scheduler.step()

    save_model(model)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-e', '--epochs', default=100)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

    