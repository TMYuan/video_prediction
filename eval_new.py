import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=256, help='number of samples')

opt = parser.parse_args()
opt.log_dir = os.path.join(opt.model_path, 'plots')
opt.model_path = os.path.join(opt.model_path, 'model.pth')
os.makedirs('%s' % opt.log_dir, exist_ok=True)

opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path)
encoder_lstm = tmp['encoder_lstm']
decoder_lstm = tmp['decoder_lstm']
content_lstm = tmp['content_lstm']
encoder_c = tmp['encoder_c']
encoder_p = tmp['encoder_p']
decoder = tmp['decoder']

encoder_lstm.eval()
decoder_lstm.eval()
content_lstm.eval()
encoder_c.eval()
encoder_p.eval()
decoder.eval()

encoder_lstm.batch_size = opt.batch_size
decoder_lstm.batch_size = opt.batch_size
content_lstm.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# --------- transfer to gpu ------------------------------------
encoder_lstm.cuda()
decoder_lstm.cuda()
content_lstm.cuda()
encoder_c.cuda()
encoder_p.cuda()
decoder.cuda()

# ---------------- set the options ----------------
opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

print(opt)

# --------- load a dataset ------------------------------------
_, test_data = utils.load_dataset(opt)

test_loader = DataLoader(test_data,
                         num_workers=opt.num_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()


# Forward functions
## Forward function for generating global content vector
def forward_content_branch(x):
    """
    Input: training data
    Return: (vec_c_seq, vec_c_global, skip connection)
    """
    vec_c_seq = [encoder_c(x[i])[0] for i in range(len(x))]
    skip = encoder_c(x[-1])[1]
    vec_c_global = content_lstm(vec_c_seq, return_last=True)[0]
    return (vec_c_seq, vec_c_global, skip)

## Forward function for encoder part in pose branch
def forward_pose_encoder_branch(x, vec_c_global, detach=False, local_only=False):
    """
    Input: training datam global content vector
    Return: (vec_p_seq, vec_p_global, mu, logvar)
    """
    vec_p_seq = [encoder_p(x[i], vec_c_global)[0] for i in range(len(x))]
    
    if local_only:
        return vec_p_seq
    else:
        if detach:
            vec_p_seq_d = [vec_p.detach() for vec_p in vec_p_seq]
        else:
            vec_p_seq_d = vec_p_seq
        vec_p_global, mu, var = encoder_lstm(vec_p_seq_d)
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


# --------- eval funtions ------------------------------------
## plot reconstructed results
def plot(x, pose_len, teacher_input=True):
    gen_seq = []
    log_cos = []
    log_mse = []
    
    _, vec_c_global, skip = forward_content_branch(x[:opt.n_past])
    vec_p_seq, vec_p_global, _, _ = forward_pose_encoder_branch(x[:pose_len], vec_c_global)
    
    # pose vector for all frames since reconstruction
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
            gen_seq.append(x[i])
            log_cos.append(F.cosine_similarity(vec_c_global.squeeze(), vec_c_global.squeeze()).data.cpu().numpy())
            log_mse.append(F.mse_loss(vec_c_global, vec_c_global).item())
        else:
            if teacher_input:
                vec_in = encoder_p(x[i - 1], vec_c_global)[0]
            else:
                vec_in = encoder_p(gen_seq[-1], vec_c_global)[0]
            vec_in = torch.cat([vec_p_global, vec_in], 1)
            vec_p_seq, hidden = decoder_lstm([vec_in], hidden=hidden)
            x_pred = decoder([torch.cat([vec_c_global, vec_p_seq[0]], 1), skip])
            gen_seq.append(x_pred)
    
    # calculate cos/mse loss for global content vector
    for i in range(opt.n_past, opt.n_eval):
        _, vec_c_global_pred, _ = forward_content_branch(x[:i])
        log_cos.append(F.cosine_similarity(vec_c_global_pred.squeeze(), vec_c_global.squeeze()).data.cpu().numpy())
        log_mse.append(F.mse_loss(vec_c_global_pred, vec_c_global).item())

        
    return gen_seq, log_cos, log_mse

def make_gifs(x, idx, name):
    recon_teacher, log_cos_1, log_mse_1 = plot(x, opt.n_eval, True)
#     print(log_cos_1)
    recon_no_teacher, log_cos_2, log_mse_2 = plot(x, opt.n_eval, False)
    
    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    all_gen = []
    all_log = []
    all_mse = []
    gt_seq = [x_i.data.cpu().numpy() for x_i in x]
    
    for s in tqdm(range(nsample), desc='sample'):
        all_gen.append([])
        all_log.append([])
        all_mse.append([])
        
        all_gen[s], all_log[s], all_mse[s] = plot(x, opt.n_past, False)
        
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(x, all_gen[s], opt.n_future)

    ###### ssim ######
    for i in range(opt.batch_size):
        gifs = [ [] for t in range(opt.n_eval) ]
        text = [ [] for t in range(opt.n_eval) ]
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(opt.n_eval):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(recon_teacher[t][i], color))
            text[t].append('Approx.\n %.4f' % (log_cos_1[t][i]))
            #posterior_2
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(recon_no_teacher[t][i], color))
            text[t].append('Approx._2\n %.4f' % (log_cos_2[t][i]))
            # best 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM\n %.4f' % (all_log[sidx][t][i]))
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random %d\n %.4f' % (s+1, all_log[s][t][i]))

        fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text, 0.5)
    return ssim, np.array(all_log), np.array(all_mse)

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

mean_ssim = 0
mean_cos = 0
mean_mse = 0
for i in range(0, opt.N, opt.batch_size):
    # plot test
    print(i)
    with torch.no_grad():
        test_x = next(testing_batch_generator)
        ssim, cos, mse = make_gifs(test_x, i, 'test')
    
    # ssim : (batch, sample, timestep)
    # cos : (sample, timestep, batch)
    # mse : (sample, timestep, batch, dim)
    mean_ssim += np.mean(ssim, axis=(0, 1))
    mean_cos += np.mean(cos, axis=(0, 2))
    mean_mse += np.mean(mse, axis=0)
mean_ssim = mean_ssim / (opt.N // opt.batch_size)
mean_cos = mean_cos / (opt.N // opt.batch_size)
mean_mse = mean_mse / (opt.N // opt.batch_size)

print(mean_ssim.shape)
print(mean_ssim)
print(mean_cos.shape)
print(mean_cos)
print(mean_mse.shape)
print(mean_mse)

