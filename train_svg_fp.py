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
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.lr = lr
    opt.pre_niter = pre_niter
    opt.pretrain = pretrain
    opt.niter = niter
    opt.epoch_size = epoch_size
    opt.batch_size = batch_size
    opt.log_dir = '%s/0328_CP_discriminator_version' % opt.log_dir
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


# ---------------- load the models  ----------------

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

import models.lstm as lstm_models
import models.discriminator as D
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    content_lstm = saved_model['content_lstm']
    posterior = saved_model['posterior']
else:
    frame_predictor = lstm_models.lstm(opt.z_dim+opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    content_lstm = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    content_lstm.apply(utils.init_weights)
    posterior.apply(utils.init_weights)

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

    
if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder_p = saved_model['encoder_p']
    encoder_c = saved_model['encoder_c']
else:
    encoder_p = model.encoder(opt.g_dim, opt.channels, conditional=True)
    encoder_c = model.encoder(opt.g_dim, opt.channels, conditional=False)
    decoder = model.decoder(opt.g_dim + opt.g_dim, opt.channels)
#     discriminator = D.discriminator(opt.g_dim)
    
    encoder_p.apply(utils.init_weights)
    encoder_c.apply(utils.init_weights)
    decoder.apply(utils.init_weights)
#     discriminator.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
content_lstm_optimizer = opt.optimizer(content_lstm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_p_optimizer = opt.optimizer(encoder_p.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_c_optimizer = opt.optimizer(encoder_c.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# initialize D
if opt.model_dir != '':
    discriminator = D.lstm(opt.g_dim, 1, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    discriminator.apply(utils.init_weights)
    discriminator_optimizer = opt.optimizer(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    discriminator.cuda()

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
bce_criterion = nn.BCELoss()
cos_criterion = nn.CosineEmbeddingLoss()
def kl_criterion(mu, logvar):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= opt.batch_size  
  return KLD

# def cos_criterion(x, y):
#     return 1 - F.cosine_similarity(x.squeeze(), y.squeeze())


# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
content_lstm.cuda()
posterior.cuda()
encoder_p.cuda()
encoder_c.cuda()
# discriminator.cuda()
decoder.cuda()
mse_criterion.cuda()
bce_criterion.cuda()
cos_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

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

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 5 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]
    # h_seq = [encoder(x[i]) for i in range(opt.n_past)]
    
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        content_lstm.hidden = content_lstm.init_hidden()
        posterior.hidden = posterior.init_hidden()
        x_in = x[0]
#         gen_seq[s].append(x_in)
        
        for i in range(opt.n_past):
            vec_c, _ = encoder_c(x[i])
            vec_c_global = content_lstm(vec_c)
        _, skip = encoder_c(x[opt.n_past - 1])
        
        for i in range(opt.n_past):
            vec_p, _ = encoder_p(x[i], vec_c_global)
            vec_p_global, mu, logvar = posterior(vec_p)
        
        for i in range(0, opt.n_eval):
#             if i <= opt.n_past:
#                 vec_c, skip = encoder_c(x[i - 1])
#             else:
#                 vec_c, _ = encoder_c(gen_seq[s][-1])
#             vec_c_global = content_lstm(vec_c)
            if i < opt.n_past:
#                 vec_p, _ = encoder_p(torch.cat(x[:opt.n_past] + [x[i]], 1))
#                 vec_p, _ = encoder_p(x[i], vec_c_global)
#                 _, vec_p_global, _ = posterior(vec_p)
                if i == 0:
                    vec_in = torch.zeros_like(vec_p)
                else:
                    vec_in, _ = encoder_p(x[i-1], vec_c_global)
                _ = frame_predictor(torch.cat([vec_p_global, vec_in], 1))
                gen_seq[s].append(x[i])
            else:
#                 vec_p_global = torch.cuda.FloatTensor(opt.batch_size, opt.z_dim).normal_()
                vec_in, _ = encoder_p(gen_seq[s][-1], vec_c_global)
                h = frame_predictor(torch.cat([vec_p_global, vec_in], 1))
#                 x_in = decoder([torch.cat([vec_c_global, h], 1), skip])
                x_in = decoder([torch.cat([vec_c_global, h], 1), skip])
                gen_seq[s].append(x_in)

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

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch) 
    utils.save_gif(fname, gifs)


def plot_rec(x, epoch):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    content_lstm.hidden = content_lstm.init_hidden()
    
    gen_seq = []
    
    for i in range(opt.n_past):
        vec_c, _ = encoder_c(x[i])
        vec_c_global = content_lstm(vec_c)
    _, skip = encoder_c(x[opt.n_past - 1])
    
    for i in range(opt.n_past+opt.n_future):
        vec_p, _ = encoder_p(x[i], vec_c_global)
        vec_p_global, mu, logvar = posterior(vec_p)
    
    
    for i in range(opt.n_past+opt.n_future):
        # Content vector
#         if i <= opt.n_past:
#             vec_c, skip = encoder_c(x[i - 1])
#         else:
#             vec_c, _ = encoder_c(x[i - 1])
#         vec_c_global = content_lstm(vec_c)
        
        # Pose vector
#         vec_p, _ = encoder_p(torch.cat(x[:opt.n_past] + [x[i]], 1))
#         vec_p, _ = encoder_p(x[i], vec_c_global)
#         _, vec_p_global, _ = posterior(vec_p)
        if i > 0:
            vec_in, _ = encoder_p(x[i - 1], vec_c_global)
        else:
            vec_in = torch.zeros_like(vec_p)
        h_pred = frame_predictor(torch.cat([vec_p_global, vec_in], 1))
        
        if i < opt.n_past:
            gen_seq.append(x[i])
        else:
            x_pred = decoder([torch.cat([vec_c_global, h_pred], 1), skip])
#             x_pred = decoder([h_pred, skip])
            gen_seq.append(x_pred)

    # forward, generate prediction
#     for i in range(1, opt.n_past+opt.n_future):
#         if i < opt.n_past:
#             gen_seq.append(x[i])
#         else:
#             h_pred = frame_predictor(vec_p_global)
#             x_pred = decoder([torch.cat([vec_c_global, h_pred], 1), skip])
#             gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)
    
def preplot_rec(x, epoch):
    content_lstm.hidden = content_lstm.init_hidden()
    
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]
    
    vec_c_seq = [encoder_c(x[i])[0] for i in range(opt.n_past)]
    _, skip = encoder_c(x[opt.n_past - 1])
    
    for i in range(opt.n_past):
        vec_c_global = content_lstm(vec_c_seq[i])

    for i in range(opt.n_past+opt.n_future):
        if i < opt.n_past:
            gen_seq.append(x[i])
        else:
            vec_p, _ = encoder_p(x[i], vec_c_global)
            x_pred = decoder([torch.cat([vec_c_global, vec_p], 1), skip]).detach()
            gen_seq.append(x_pred)
            
    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/pre_rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

# --------- swap loss function -------------------
def swap_cp(x):
    """
    Return mse after swapping content and pose
    """
    swap_mse = 0
    
    frame_predictor.hidden = frame_predictor.init_hidden()
    content_lstm.hidden = content_lstm.init_hidden()
    posterior.hidden = posterior.init_hidden()
    
    vec_c_seq = [encoder_c(x[i])[0] for i in range(opt.n_past, opt.n_past+opt.n_future)]
    _, skip = encoder_c(x[opt.n_past + opt.n_future - 1])
    for i in range(opt.n_future):
        vec_c_global = content_lstm(vec_c_seq[i])
        
    for i in range(opt.n_past):
        vec_p, _ = encoder_p(x[i], vec_c_global)
        vec_p_global, _, _ = posterior(vec_p)
        h_pred = frame_predictor(vec_p_global)
        x_pred = decoder([torch.cat([vec_c_global, h_pred], 1), skip])
        swap_mse += mse_criterion(x_pred, x[i])
    return swap_mse

# --------- train discriminator --------------------------------------
def train_D(x):
    """
    Training function for discriminator
    """
    discriminator.zero_grad()
    
    discriminator.hidden = discriminator.init_hidden()
    content_lstm.hidden = content_lstm.init_hidden()
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    
    loss_d = 0
    
    real_vec_c = [encoder_c(x[i])[0].detach() for i in range(opt.n_past+opt.n_future)]
    
    for i in range(opt.n_past):
        vec_c_global = content_lstm(real_vec_c[i]).detach()
    vec_c_fix = vec_c_global.detach()
    _, skip = encoder_c(x[opt.n_past - 1])
    skip_fix = [s.detach() for s in skip]
    
    # re-generate global pose vector
    vec_p_seq = [encoder_p(x[i], vec_c_fix)[0] for i in range(opt.n_past+opt.n_future)]
    for i in range(opt.n_past+opt.n_future):
        vec_p = vec_p_seq[i].detach()
        vec_p_global, mu, logvar = posterior(vec_p)
        
    # content preserve loss
    fake_vec_c = []
    for i in range(opt.n_past+opt.n_future):
        if i > 0:
            vec_in = vec_p_seq[i - 1].detach()
        else:
            vec_in = torch.zeros_like(vec_p).detach()
        vec_p_recon = frame_predictor(torch.cat([vec_p_global, vec_in], 1)).detach()
        
        x_pred = decoder([torch.cat([vec_c_fix, vec_p_recon], 1), skip_fix])
        vec_c, _ = encoder_c(x_pred)
        fake_vec_c.append(vec_c.detach())
    
    # forward in D
    discriminator.hidden = discriminator.init_hidden()
    for r in real_vec_c:
        real_output = discriminator(r)
    
    discriminator.hidden = discriminator.init_hidden()
    for f in fake_vec_c:
        fake_output = discriminator(f)
    
    loss_d = -torch.mean(real_output) + torch.mean(fake_output)
    loss_d.backward()
    discriminator_optimizer.step()
    
    # Clip weights of discriminator
    for p in discriminator.parameters():
        p.data.clamp_(-opt.clip_value, opt.clip_value)
        
    return loss_d.data.cpu().numpy()
    
# --------- train deterministic part funtions ------------------------------------
def train_deterministic(x):
    """
    Train content encoder, content LSTM, pose encoder(without LSTM), decoder
    """
    # zero_grad and initialize the hidden state.
    content_lstm.zero_grad()
    encoder_p.zero_grad()
    encoder_c.zero_grad()
    decoder.zero_grad()
    content_lstm.hidden = content_lstm.init_hidden()
    
    # log variable
    mse = 0
    preserve_loss = 0
    swap_mse = 0
    
    # generate content vector in each time step and produce global content vector
    vec_c_seq = [encoder_c(x[i])[0] for i in range(opt.n_past)]
    _, skip = encoder_c(x[opt.n_past - 1])
    for i in range(opt.n_past):
        vec_c_global = content_lstm(vec_c_seq[i])

    # generate pose vector for each time step without temporary
    for i in range(opt.n_past+opt.n_future):
        vec_p, _ = encoder_p(x[i], vec_c_global)
        x_pred = decoder([torch.cat([vec_c_global, vec_p], 1), skip])
        mse += mse_criterion(x_pred, x[i])

    loss = mse
    loss.backward()
    
    content_lstm_optimizer.step()
    encoder_c_optimizer.step()
    encoder_p_optimizer.step()
    decoder_optimizer.step()
    
    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future)

# --------- train pose vae lstm funtions ------------------------------------
def train_posevae(x):
    """
    Train pose encoder LSTM and pose decoder LSTM
    (frame_predictor -> pose decoder LSTM)
    (posterior -> pose encoder LSTM)
    """
    # zero_grad and initialize the hidden state.
    frame_predictor.zero_grad()
    posterior.zero_grad()
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()

    # log variable: pose_recon and KLD
    kld = 0
    pose_recon = 0
    
    # generate content vector in each time step and produce global content vector
    vec_c_seq = [encoder_c(x[i])[0] for i in range(opt.n_past)]
    for i in range(opt.n_past):
        vec_c_global = content_lstm(vec_c_seq[i])
    vec_c_global = vec_c_global.detach()
    
    _, skip = encoder_c(x[opt.n_past - 1])
    skip = [s.detach() for s in skip]
    
    # Generate pose vector and prediction for each time step
    vec_p_seq = [encoder_p(x[i], vec_c_global)[0] for i in range(opt.n_past+opt.n_future)]
    
    # KLD loss for pose vae
    for i in range(opt.n_past+opt.n_future):
        vec_p = vec_p_seq[i].detach()
        vec_p_global, mu, logvar = posterior(vec_p)
    kld += kl_criterion(mu, logvar)
    
    # train decoder part(pose lstm decoder & frame decoder)
    for i in range(opt.n_past+opt.n_future):
        if i > 0:
            vec_in = vec_p_seq[i - 1].detach()
        else:
            vec_in = torch.zeros_like(vec_p)
        vec_p_recon = frame_predictor(torch.cat([vec_p_global, vec_in], 1))
        vec_p_target = vec_p_seq[i].detach()
        
        pose_recon += mse_criterion(vec_p_recon, vec_p_target)

    loss = pose_recon + kld*opt.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()

    return kld.data.cpu().numpy(), pose_recon.data.cpu().numpy()/(opt.n_past+opt.n_future)

# Train the overall model function
def train_overall(x):
    """
    Train overall model:
    encoder_c, content_lstm, encoder_p, frame_predictor, posterior, decoder
    """
    # zero_grad and initialize the hidden state.
    content_lstm.zero_grad()
    encoder_p.zero_grad()
    encoder_c.zero_grad()
    decoder.zero_grad()
    frame_predictor.zero_grad()
    posterior.zero_grad()
    
    content_lstm.hidden = content_lstm.init_hidden()
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    
    # log variable: pose_recon and KLD
    mse = 0
    kld = 0
    pose_recon = 0
    preserve = 0
    loss_G = 0
    
    # generate content vector in each time step and produce global content vector
    vec_c_seq = [encoder_c(x[i])[0] for i in range(opt.n_past)]
    for i in range(opt.n_past):
        vec_c_global = content_lstm(vec_c_seq[i])
    _, skip = encoder_c(x[opt.n_past - 1])
    
    # Generate pose vector and prediction for each time step
    vec_p_seq = [encoder_p(x[i], vec_c_global)[0] for i in range(opt.n_past+opt.n_future)]
    
    # KLD loss for pose vae(train pose lstm)
    for i in range(opt.n_past+opt.n_future):
        vec_p = vec_p_seq[i].detach()
        vec_p_global, mu, logvar = posterior(vec_p)
    kld += kl_criterion(mu, logvar)
    
    # for pose reconstruction loss (only train pose lstm, fix pose encoder)
    frame_predictor.hidden = frame_predictor.init_hidden()
    for i in range(opt.n_past+opt.n_future):
        if i > 0:
            vec_in = vec_p_seq[i - 1].detach()
        else:
            vec_in = torch.zeros_like(vec_p)
        vec_p_recon = frame_predictor(torch.cat([vec_p_global, vec_in], 1))
        vec_p_target = vec_p_seq[i].detach()
        
        pose_recon += mse_criterion(vec_p_recon, vec_p_target)
    
    # for image reconstruction loss (train all model)
    frame_predictor.hidden = frame_predictor.init_hidden()
    discriminator.hidden = discriminator.init_hidden()
    for i in range(opt.n_past+opt.n_future):
        if i > 0:
            vec_in = vec_p_seq[i - 1]
        else:
            vec_in = torch.zeros_like(vec_p)
        vec_p_recon = frame_predictor(torch.cat([vec_p_global, vec_in], 1))
        x_pred = decoder([torch.cat([vec_c_global, vec_p_recon], 1), skip])
        
        # Input of D
        vec_c_G = encoder_c(x_pred)[0]
        # only take out final result
        loss_G = -torch.mean(discriminator(vec_c_G))
        
        mse += mse_criterion(x_pred, x[i])
    
    # content preserve loss
#     loss_G = train_G(x)
#     preserve_loss = 0.1 * preserve
    
    # backward
#     loss = mse + pose_recon + kld*opt.beta + preserve
    loss = mse + pose_recon + kld*opt.beta + loss_G
    
#     loss.backward(retain_graph=True)
    loss.backward()
    content_lstm_optimizer.step()
    encoder_c_optimizer.step()
    
#     preserve.backward()
    encoder_p_optimizer.step()
    decoder_optimizer.step()
    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    
#     return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy(), pose_recon.data.cpu().numpy()/(opt.n_past+opt.n_future), preserve.data.cpu().numpy()/(opt.n_future)
    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy(), pose_recon.data.cpu().numpy()/(opt.n_past+opt.n_future), loss_G.data.cpu().numpy()

# content preservation (train decoder, pose branch; fix content branch)
def train_preserve(x):
    #log variable
    preserve = 0
    
    content_lstm.hidden = content_lstm.init_hidden()
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    
    # generate content vector in each time step and produce global content vector
    vec_c_seq = [encoder_c(x[i])[0] for i in range(opt.n_past)]
    for i in range(opt.n_past):
        vec_c_global = content_lstm(vec_c_seq[i])
    vec_c_fix = vec_c_global.detach()
    _, skip = encoder_c(x[opt.n_past - 1])
    skip_fix = [s.detach() for s in skip]
    
    # re-generate global pose vector
    vec_p_seq = [encoder_p(x[i], vec_c_fix)[0] for i in range(opt.n_past+opt.n_future)]
    for i in range(opt.n_past+opt.n_future):
        vec_p = vec_p_seq[i].detach()
        vec_p_global, mu, logvar = posterior(vec_p)
    
    # content preserve loss
    for i in range(opt.n_past+opt.n_future):
        if i > 0:
            vec_in = vec_p_seq[i - 1]
        else:
            vec_in = torch.zeros_like(vec_p)
        vec_p_recon = frame_predictor(torch.cat([vec_p_global, vec_in], 1))
        
        if i >= opt.n_past:
            x_pred = decoder([torch.cat([vec_c_fix, vec_p_recon], 1), skip_fix])
            vec_c, _ = encoder_c(x_pred)
            vec_c_pred = content_lstm(vec_c)
            preserve += mse_criterion(vec_c_pred, vec_c_fix)
            
    return preserve

# --------- Pre-train loop -----------------------------------
# Train the model without temporary on pose encoder & variation
if opt.pretrain:
    for epoch in tqdm(range(opt.pre_niter), desc='[PRE]EPOCH'):
        content_lstm.train()
        encoder_c.train()
        encoder_p.train()
        decoder.train()
        posterior.train()
        frame_predictor.train()
        
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
    
        # Make visualization
        content_lstm.eval()
        encoder_c.eval()
        encoder_p.eval()
        decoder.eval()
        posterior.eval()
        frame_predictor.eval()
        
        x = next(testing_batch_generator)
        with torch.no_grad():
            plot_rec(x, epoch)
            plot(x, epoch)
            
        if (epoch+1) % 10 == 0:
            torch.save({
                'encoder_p': encoder_p,
                'encoder_c': encoder_c,
                'decoder': decoder,
                'content_lstm': content_lstm,
                'posterior' : posterior,
                'frame_predictor' : frame_predictor,
                'opt': opt},
                '%s/pre_model.pth' % opt.log_dir)
    

# --------- training loop ------------------------------------
for epoch in tqdm(range(opt.niter), desc='EPOCH'):
    if (epoch+1) % 50 == 0:
        opt.beta *= 10
    
    frame_predictor.train()
    content_lstm.train()
    posterior.train()
    encoder_c.train()
    encoder_p.train()
    decoder.train()
    
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

    # plot some stuff
    frame_predictor.eval()
    content_lstm.eval()
    encoder_c.eval()
    encoder_p.eval()
    decoder.eval()
    posterior.eval()
    x = next(testing_batch_generator)
    with torch.no_grad():
        plot(x, epoch)
        plot_rec(x, epoch)

    # save the model
    torch.save({
        'encoder_p': encoder_p,
        'encoder_c': encoder_c,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'content_lstm': content_lstm,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
        

