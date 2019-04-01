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
    decoder_lstm = saved_model['decoder_lstm']
    content_lstm = saved_model['content_lstm']
    encoder_lstm = saved_model['encoder_lstm']
    decoder = saved_model['decoder']
    encoder_p = saved_model['encoder_p']
    encoder_c = saved_model['encoder_c']
else:
    decoder_lstm = lstm_models.lstm_new(opt.z_dim+opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    content_lstm = lstm_models.lstm_new(opt.g_dim, opt.g_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    encoder_lstm = lstm_models.gaussian_lstm_new(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    encoder_p = model.encoder(opt.g_dim, opt.channels, conditional=True)
    encoder_c = model.encoder(opt.g_dim, opt.channels, conditional=False)
    decoder = model.decoder(opt.g_dim + opt.g_dim, opt.channels)
    
    decoder_lstm.apply(utils.init_weights)
    content_lstm.apply(utils.init_weights)
    encoder_lstm.apply(utils.init_weights)
    encoder_p.apply(utils.init_weights)
    encoder_c.apply(utils.init_weights)
    decoder.apply(utils.init_weights)
    
## Create Optimizer
decoder_lstm_optimizer = opt.optimizer(decoder_lstm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
content_lstm_optimizer = opt.optimizer(content_lstm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_lstm_optimizer = opt.optimizer(encoder_lstm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_p_optimizer = opt.optimizer(encoder_p.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_c_optimizer = opt.optimizer(encoder_c.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


discriminator = D.lstm_new(opt.g_dim, 1, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
discriminator.apply(utils.init_weights)
discriminator_optimizer = opt.optimizer(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
discriminator.cuda()
    
## Loss function
mse_criterion = nn.MSELoss()
bce_criterion = nn.BCELoss()
def kl_criterion(mu, logvar):
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= opt.batch_size  
    return kld

## To GPU
decoder_lstm.cuda()
content_lstm.cuda()
encoder_lstm.cuda()
encoder_p.cuda()
encoder_c.cuda()
# discriminator.cuda()
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
def forward_content_branch(x):
    """
    Input: training data
    Return: (vec_c_seq, vec_c_global, skip connection)
    """
    vec_c_seq = [encoder_c(x[i])[0] for i in range(len(x))]
    skip = encoder_c(x[opt.n_past - 1])[1]
    vec_c_global = content_lstm(vec_c_seq, return_last=True)[0]
    return (vec_c_seq, vec_c_global, skip)

## Forward function for encoder part in pose branch
def forward_pose_encoder_branch(x, vec_c_global, detach=False, local_only=False):
    """
    Input: training datam global content vector
    Return: (vec_p_seq, vec_p_global, mu, logvar)
    """
    vec_p_seq = [encoder_p(x[i], vec_c_global)[0] for i in range(len(x))]
    
    if detach:
        vec_p_seq = [vec_p.detach() for vec_p in vec_p_seq]
    
    if local_only:
        return vec_p_seq
    else:
        vec_p_global, mu, var = encoder_lstm(vec_p_seq)
        return (vec_p_seq, vec_p_global, mu, var)
    
## Forward function for decoder part in pose branch
def forward_pose_decoder_branch(vec_p_seq, vec_p_global):
    """
    Input: pose vector sequence, global pose vector
    Return: resonstructed pose vector sequence
    """
    vec_in_seq = []
    for i in range(len(vec_p_seq)):
        if i > 0:
            vec_in = vec_p_seq[i - 1]
        else:
            vec_in = torch.zeros_like(vec_p_seq[0])
        vec_in_seq.append(torch.cat([vec_p_global, vec_in], 1))
    vec_p_recon_seq = decoder_lstm(vec_in_seq)[0]
    return vec_p_recon_seq
    
## Forward function for decoder
def forward_decoder(vec_p_seq, vec_c_global, skip):
    """
    Input: pose vec sequence, global content vector, skip connection
    Return: list of predicted frames
    """
    x_pred_list = []
    for i in range(len(vec_p_seq)):
        vec_p = vec_p_seq[i]
        x_pred = decoder([torch.cat([vec_c_global, vec_p], 1), skip])
        x_pred_list.append(x_pred)
    return x_pred_list

## Forward function for discriminator
def forward_discriminator(x):
    """
    Input: training data
    Return: result of discriminator
    """
    vec_c_seq = [encoder_c(x[i])[0] for i in range(len(x))]
    out_D = discriminator(vec_c_seq, return_last=True)[0]
    return out_D
    
## train deterministic part funtions
def train_deterministic(x):
    """
    Train content encoder, content LSTM, pose encoder(without LSTM), decoder
    """
    # zero_grad and initialize the hidden state.
    content_lstm.zero_grad()
    encoder_p.zero_grad()
    encoder_c.zero_grad()
    decoder.zero_grad()
    
    # log variable
    mse = 0
    preserve_loss = 0
    swap_mse = 0
    
    # generate content vector in each time step and produce global content vector
    vec_c_seq, vec_c_global, skip = forward_content_branch(x[:opt.n_past])
    
    # generate pose vector in each time step
    vec_p_seq = forward_pose_encoder_branch(x, vec_c_global, local_only=True)
    x_pred_list = forward_decoder(vec_p_seq, vec_c_global, skip)
    
    # Calculate loss
    for (pred, gt) in zip(x_pred_list, x):
        mse += mse_criterion(pred, gt)
        
    # Backward
    loss = mse
    loss.backward()
    
    content_lstm_optimizer.step()
    encoder_c_optimizer.step()
    encoder_p_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / (opt.n_past+opt.n_future)

## train pose vae lstm funtions
def train_posevae(x):
    """
    Train pose encoder LSTM and pose decoder LSTM
    """
    # zero_grad and initialize the hidden state.
    decoder_lstm.zero_grad()
    encoder_lstm.zero_grad()
    
    # log variable: pose_recon and KLD
    kld = 0
    pose_recon = 0
    
    # generate content vector in each time step and produce global content vector
    vec_c_seq, vec_c_global, skip = forward_content_branch(x[:opt.n_past])
    vec_c_global = vec_c_global.detach()
    skip = [s.detach() for s in skip]
    
    # Generate pose vector for each time step
    vec_p_seq, vec_p_global, mu, var = forward_pose_encoder_branch(x, vec_c_global, detach=True)
    kld += kl_criterion(mu, var)
    
    # train decoder part(pose lstm decoder & frame decoder)
    vec_p_recon_seq = forward_pose_decoder_branch(vec_p_seq, vec_p_global)
    
    # calculate pose reconstruction loss
    for (recon, target) in zip(vec_p_recon_seq, vec_p_seq):
        pose_recon += mse_criterion(recon, target)
        
    loss = pose_recon + opt.beta*kld
    loss.backward()
    
    decoder_lstm_optimizer.step()
    encoder_lstm_optimizer.step()
    
    return kld.item(), pose_recon.item() / (opt.n_past+opt.n_future)

## Train the overall model function
def train_overall(x):
    """
    Train overall model:
    encoder_c, content_lstm, encoder_p, encoder_lstm, decoder_lstm, decoder
    """
    # zero_grad and initialize the hidden state.
    content_lstm.zero_grad()
    encoder_p.zero_grad()
    encoder_c.zero_grad()
    decoder.zero_grad()
    decoder_lstm.zero_grad()
    encoder_lstm.zero_grad()
    
    # log variable: pose_recon and KLD
    mse = 0
    kld = 0
    pose_recon = 0
    preserve = 0
    loss_G = 0
    
    # generate content vector in each time step and produce global content vector
    vec_c_seq, vec_c_global, skip = forward_content_branch(x[:opt.n_past])
    
    # KLD loss for pose vae
    vec_p_seq, vec_p_global, mu, var = forward_pose_encoder_branch(x, vec_c_global)
    kld += kl_criterion(mu, var)
    
    # pose reconstruction loss
    vec_p_recon_seq = forward_pose_decoder_branch(vec_p_seq, vec_p_global)
    for (recon, target) in zip(vec_p_recon_seq, vec_p_seq):
        pose_recon += mse_criterion(recon, target)

    # image reconstruction loss
    x_pred_list = forward_decoder(vec_p_recon_seq, vec_c_global, skip)
    for (pred, gt) in zip(x_pred_list, x):
        mse += mse_criterion(pred, gt)
        
    # adversarial loss for generator
    out_D = forward_discriminator(x_pred_list)
    loss_G = -torch.mean(out_D)
    
    loss_all_model = mse + loss_G
    loss_pose_vae = pose_recon + opt.beta*kld
    
    loss_all_model.backward(retain_graph=True)
    content_lstm_optimizer.step()
    encoder_c_optimizer.step()
    encoder_p_optimizer.step()
    decoder_optimizer.step()
    
    loss_pose_vae.backward()
    encoder_lstm_optimizer.step()
    decoder_lstm_optimizer.step()
    
    return mse.item()/(opt.n_past+opt.n_future), kld.item(), pose_recon.item()/(opt.n_past+opt.n_future), loss_G.item()

def train_D(x):
    """
    Training function for discriminator
    """
    discriminator.zero_grad()
    
    loss_d = 0
    
    with torch.no_grad():
        # generate content vector in each time step and produce global content vector
        vec_c_seq, vec_c_global, skip = forward_content_branch(x[:opt.n_past])
        vec_p_seq, vec_p_global, mu, var = forward_pose_encoder_branch(x, vec_c_global)
        vec_p_recon_seq = forward_pose_decoder_branch(vec_p_seq, vec_p_global)
        x_pred_list = forward_decoder(vec_p_recon_seq, vec_c_global, skip)
    
    x_pred_list = [x_pred.detach() for x_pred in x_pred_list]
    real_output = forward_discriminator(x)
    fake_output = forward_discriminator(x_pred_list)
    
    loss_d = -torch.mean(real_output) + torch.mean(fake_output)
    loss_d.backward()
    discriminator_optimizer.step()
    
    # Clip weights of discriminator
    for p in discriminator.parameters():
        p.data.clamp_(-opt.clip_value, opt.clip_value)
        
    return loss_d.item()

# Plot functions
## plot reconstructed results
def plot_rec(x, epoch, prefix=''):
    gen_seq = []
    vec_c_seq, vec_c_global, skip = forward_content_branch(x[:opt.n_past])
    vec_p_seq, vec_p_global, _, _ = forward_pose_encoder_branch(x, vec_c_global)
    vec_p_recon_seq = forward_pose_decoder_branch(vec_p_seq, vec_p_global)
    x_pred_list = forward_decoder(vec_p_recon_seq, vec_c_global, skip)
    gen_seq += x[:opt.n_past]
    gen_seq += x_pred_list[opt.n_past:]
    
    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/[%s]rec_%d.png' % (opt.log_dir, prefix, epoch) 
    utils.save_tensors_image(fname, to_plot)
    
## plot sampling results
def plot(x, epoch, prefix=''):
    nsample = 5
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]
    
    for s in range(nsample):
        _, vec_c_global, skip = forward_content_branch(x[:opt.n_past])
        vec_p_seq, vec_p_global, _, _ = forward_pose_encoder_branch(x[:opt.n_past], vec_c_global)
        
        # pose vector for current frames
        vec_in_seq = []
        for i in range(opt.n_past):
            if i > 0:
                vec_in = vec_p_seq[i - 1]
            else:
                vec_in = torch.zeros_like(vec_p_seq[0])
            vec_in_seq.append(torch.cat([vec_p_global, vec_in], 1))
        hidden = decoder_lstm(vec_in_seq)[1]
        
        # predict
        for i in range(opt.n_eval):
            if i < opt.n_past:
                gen_seq[s].append(x[i])
            else:
                vec_in = encoder_p(gen_seq[s][-1], vec_c_global)[0]
                vec_in = torch.cat([vec_p_global, vec_in], 1)
                vec_p_seq, hidden = decoder_lstm([vec_in])
                x_pred = decoder([torch.cat([vec_c_global, vec_p_seq[0]], 1), skip])
                gen_seq[s].append(x_pred)

    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [] 
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        for s in range(nsample):
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for s in range(nsample):
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/[%s]sample_%d.png' % (opt.log_dir, prefix, epoch) 
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/[%s]sample_%d.gif' % (opt.log_dir, prefix, epoch) 
    utils.save_gif(fname, gifs)

# Training loop
def pre_training_loop():
    for epoch in tqdm(range(opt.pre_niter), desc='[PRE]EPOCH'):
        content_lstm.train()
        encoder_c.train()
        encoder_p.train()
        decoder.train()
        encoder_lstm.train()
        decoder_lstm.train()
        
        epoch_mse = 0
        epoch_kld = 0
        epoch_pose_recon = 0
        
        for i in tqdm(range(opt.epoch_size), desc='[PRE]BATCH'):
            x = next(training_batch_generator)
            mse = train_deterministic(x)
            kld, pose_recon = train_posevae(x)
            
            epoch_mse += mse
            epoch_kld += kld
            epoch_pose_recon += pose_recon
            
        print('[PRE][%02d] mse loss: %.5f | kld loss: %.5f | pose recon loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch_pose_recon/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
        
        
        if (epoch+1) % 10 == 0:
            """
            Make visualization
            """
            content_lstm.eval()
            encoder_c.eval()
            encoder_p.eval()
            decoder.eval()
            encoder_lstm.eval()
            decoder_lstm.eval()

            x = next(testing_batch_generator)
            with torch.no_grad():
                plot_rec(x, epoch, 'PRE')
                plot(x, epoch, 'PRE')
            torch.save({
                'encoder_p': encoder_p,
                'encoder_c': encoder_c,
                'decoder': decoder,
                'content_lstm': content_lstm,
                'encoder_lstm' : encoder_lstm,
                'decoder_lstm' : decoder_lstm,
                'opt': opt
            }, '%s/pre_model.pth' % opt.log_dir)
            
def training_loop():
    for epoch in tqdm(range(opt.niter), desc='EPOCH'):
        if (epoch+1) % 50 == 0:
            opt.beta *= 10

        content_lstm.train()
        encoder_c.train()
        encoder_p.train()
        decoder.train()
        encoder_lstm.train()
        decoder_lstm.train()
        discriminator.train()

        epoch_mse = 0
        epoch_kld = 0
        epoch_pose_recon = 0
        epoch_preserve = 0
        epoch_pose_recon = 0
        epoch_swap_mse = 0
        epoch_loss_d = 0
        epoch_loss_g = 0

        for i in tqdm(range(opt.epoch_size), desc='BATCH'):
            x = next(training_batch_generator)

            # train disc.
            epoch_loss_d += train_D(x)

            # train generator
            mse, kld, pose_recon, loss_g = train_overall(x)
            epoch_mse += mse
            epoch_kld += kld
            epoch_pose_recon += pose_recon
            epoch_loss_g += loss_g


        print('[%02d] mse loss: %.5f | kld loss: %.5f | pose recon loss: %.5f | loss_d: %.5f | loss_g: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch_pose_recon/opt.epoch_size, epoch_loss_d/opt.epoch_size, epoch_loss_g/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
        
        """
        Make Visualization
        """
        content_lstm.eval()
        encoder_c.eval()
        encoder_p.eval()
        decoder.eval()
        encoder_lstm.eval()
        decoder_lstm.eval()
        
        x = next(testing_batch_generator)
        with torch.no_grad():
            plot_rec(x, epoch, 'NORMAL')
            plot(x, epoch, 'NORMAL')
        # save the model
        torch.save({
            'encoder_p': encoder_p,
            'encoder_c': encoder_c,
            'decoder': decoder,
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'content_lstm': content_lstm,
            'opt': opt
        }, '%s/model.pth' % opt.log_dir)
        if epoch % 10 == 0:
            print('log dir: %s' % opt.log_dir)
            

pre_training_loop()
training_loop()