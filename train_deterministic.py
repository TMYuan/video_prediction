from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import utils
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/fp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--pre_niter', type=int, default=50, help='number of epochs to pre-train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=1e-4, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--content_model', default='cnnrnn', help='model type (convlstm | cnnrnn)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--pretrain', action='store_true', help='if true, train model without teporary first')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--threshold', type=float, default=0.9, help='ratio of teacher input')

opt = parser.parse_args()
if opt.model_dir != '':
    saved_model = torch.load('%s/pre_model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    lr = opt.lr
    pretrain = opt.pretrain
    pre_niter = opt.pre_niter
    niter = opt.niter
    epoch_size = opt.epoch_size
    batch_size = opt.batch_size
    log_dir = opt.log_dir
    threshold = opt.threshold
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.lr = lr
    opt.pre_niter = pre_niter
    opt.pretrain = pretrain
    opt.niter = niter
    opt.epoch_size = epoch_size
    opt.batch_size = batch_size
    opt.log_dir = log_dir
    opt.threshold = threshold
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%d-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)
    
# model definition
import models.lstm as lstm_models
import models.discriminator as D
if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
    
## Load/Create model
if opt.model_dir != '':
    print('Load Pretrain Model')
    decoder = saved_model['decoder']
    encoder_p = saved_model['encoder_p']
    encoder_c = saved_model['encoder_c']
else:
    print('Train the model from scratch')
    encoder_p = model.encoder(opt.g_dim, opt.channels, conditional=True)
    encoder_c = model.encoder(opt.g_dim, opt.channels, conditional=False)
    decoder = model.decoder(opt.g_dim + opt.g_dim, opt.channels)
    
    encoder_p.apply(utils.init_weights)
    encoder_c.apply(utils.init_weights)
    decoder.apply(utils.init_weights)
    
## Create Optimizer
encoder_p_optimizer = opt.optimizer(encoder_p.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_c_optimizer = opt.optimizer(encoder_c.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

"""
TODO: Change structure of discriminator
"""
discriminator = D.discriminator(opt.g_dim)
discriminator.apply(utils.init_weights)
discriminator_optimizer = opt.optimizer(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
discriminator.cuda()

## Loss function
mse_criterion = nn.MSELoss()
bce_criterion = nn.BCELoss()

## To GPU
encoder_p.cuda()
encoder_c.cuda()
decoder.cuda()
mse_criterion.cuda()
bce_criterion.cuda()

## Load dataset
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(
    train_data,
    num_workers=opt.data_threads,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)
test_loader = DataLoader(
    test_data,
    num_workers=opt.data_threads,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()


# Training function

## Forward function for generating global content vector
def forward_content_encoder(x):
    """
    Input: training data
    Return: (vec_c_seq, vec_c_global, skip connection)
    """
    vec_c_seq = [encoder_c(x[i])[0] for i in range(len(x))]
    skip_seq = [encoder_c(x[i])[1] for i in range(len(x))]
    
    return vec_c_seq, skip_seq

## Forward function for encoder part in pose branch
def forward_pose_encoder(x, vec_c_seq):
    """
    Input: x and content vector seq
    Return: pose vector sequence
    """
    vec_p_seq = [encoder_p(x[i], vec_c_seq[i])[0] for i in range(len(x))]
    
    return vec_p_seq
    
## Forward function for decoder
def forward_decoder(vec_p_seq, vec_c_seq, skip_seq):
    """
    Input: pose vec sequence, content vec sequence, skip sequence
    Return: list of predicted frames
    """
    x_pred_list = []
    for vec_p, vec_c, skip in zip(vec_p_seq, vec_c_seq, skip_seq):
        x_pred = decoder([torch.cat([vec_c, vec_p], 1), skip])
        x_pred_list.append(x_pred)
    
    return x_pred_list

## train deterministic part funtions
def train_deterministic(x):
    """
    Train content encoder, pose encoder(without LSTM), decoder, discriminator
    """
    # zero_grad
    encoder_p.zero_grad()
    encoder_c.zero_grad()
    decoder.zero_grad()
    discriminator.zero_grad()
    
    # log variable
    mse = 0
    loss_G = 0
    
    # generate content vector seq and skip seq
    vec_c_seq, skip_seq = forward_content_encoder(x)
    
    # generate pose vector in each time step
    vec_p_seq = forward_pose_encoder(x, vec_c_seq)
    
    # reconstruction
    x_pred_list = forward_decoder(vec_p_seq, vec_c_seq, skip_seq)
    
    # reconstruction loss
    for (pred, gt) in zip(x_pred_list, x):
        mse += mse_criterion(pred, gt)
        
    # Swap the training data to generate different pose
    rp = torch.randperm(opt.batch_size).cuda()
    vec_p_swap = []
    for vec_p in vec_p_seq:
        vec_swap = torch.zeros_like(vec_p)
        vec_swap.copy_(vec_p)
        vec_swap = vec_swap[rp]