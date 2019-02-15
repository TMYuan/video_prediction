import torch
import torch.optim as optim
import torch.nn as nn
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
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
content_lstm = tmp['content_lstm']
encoder_c = tmp['encoder_c']
encoder_p = tmp['encoder_p']
decoder = tmp['decoder']

frame_predictor.eval()
posterior.eval()
content_lstm.eval()
encoder_c.eval()
encoder_p.eval()
decoder.eval()

frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
content_lstm.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
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

# --------- eval funtions ------------------------------------
def make_gifs(x, idx, name):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    content_lstm.hidden = content_lstm.init_hidden()
    posterior_gen = []
    x_in = x[0]
    
    vec_c_seq = [encoder_c(x[i])[0] for i in range(opt.n_past)]
    _, skip = encoder_c(x[opt.n_past - 1])
    for i in range(opt.n_past):
        vec_c_global = content_lstm(vec_c_seq[i])
    
    for i in range(0, opt.n_eval):
        vec_p, _ = encoder_p(x[i])
        _, vec_p_global, _ = posterior(vec_p)
        if i < opt.n_past:
            posterior_gen.append(x[i])
        else:
            h_pred = frame_predictor(vec_p_global).detach()
            x_pred = decoder([torch.cat([vec_c_global, h_pred], 1), skip])
            posterior_gen.append(x_pred)
  
    # random sample
    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    all_gen = []
    for s in tqdm(range(nsample), desc='sample'):
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        content_lstm.hidden = content_lstm.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        
        vec_c_seq = [encoder_c(x[i])[0] for i in range(opt.n_past)]
        _, skip = encoder_c(x[opt.n_past - 1])
        for i in range(opt.n_past):
            vec_c_global = content_lstm(vec_c_seq[i])
        
        for i in range(1, opt.n_eval):
            if i < opt.n_past:
                all_gen[s].append(x[i])
            else:
                vec_p_global = torch.cuda.FloatTensor(opt.batch_size, opt.z_dim).normal_()
                h_pred = frame_predictor(vec_p_global).detach()
                x_pred = decoder([torch.cat([vec_c_global, h_pred], 1), skip])
                gen_seq.append(x_pred.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_pred)
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)


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
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text)
    return ssim

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
for i in range(0, opt.N, opt.batch_size):
    # plot test
    print(i)
    with torch.no_grad():
        test_x = next(testing_batch_generator)
        ssim = make_gifs(test_x, i, 'test')
    mean_ssim += ssim.mean()
mean_ssim = mean_ssim / (opt.N // opt.batch_size)

print(mean_ssim)
    
