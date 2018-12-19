import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import shutil
import numpy as np
import torch
from torch.backends import cudnn
import torch.optim as optim
import torch.nn as nn

from other.load_data import LoadDataset

# from src.util import vae_loss, calc_gradient_penalty, interpolate_vae_3d

from tensorboardX import SummaryWriter
cudnn.benchmark = True

exp_name='CVAE'
checkpoint_dir='checkpoint/'  # Folder for model checkpoints
log_dir='log/'  # Folder for training logs
data_root='/home/zhangjingqiu/data_total/UTKFace/'
seed=1
img_size=64
img_depth=3
domain_a='1'
doamin_b='4'
doamin_c='7'
batch_size=64

enc_dim = 1024
code_dim = 3
vae_learning_rate = 0.0001
vae_betas = [0.5,0.999]
# df_learning_rate = conf['model']['D_feat']['lr']
# df_betas = tuple(conf['model']['D_feat']['betas'])
dp_learning_rate = 0.0001
dp_betas =[0.5,0.999]

model_path = checkpoint_dir + exp_name + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_path = model_path + '{}'

if os.path.exists(log_dir + exp_name):
    shutil.rmtree(log_dir + exp_name)
writer = SummaryWriter(log_dir + exp_name)

#load dataset
np.random.seed(seed)

a_loader = LoadDataset('face',data_root,batch_size,'train',style=domain_a)
b_loader = LoadDataset('face',data_root,batch_size,'train',style=doamin_b)
c_loader = LoadDataset('face',data_root,batch_size,'train',style=doamin_c)

a_test = LoadDataset('face',data_root,1,'test',style=domain_a)
b_test = LoadDataset('face',data_root,1,'test',style=doamin_b)
c_test = LoadDataset('face',data_root,1,'test',style=doamin_c)

for d1,d2,d3 in zip(a_test,b_test,c_test):
    a_test_sample = d1[0].type(torch.FloatTensor)
    b_test_sample = d2[0].type(torch.FloatTensor)
    c_test_sample = d3[0].type(torch.FloatTensor)
    break

CVAE=CVAE(enc_dim,code_dim,img_depth)
D_pix=Discriminator_pixel(img_depth)
reconstruct_loss = torch.nn.MSELoss()
clf_loss = nn.BCEWithLogitsLoss()

vae=CVAE.cuda()
d_pix=D_pix.cuda()
reconstruct_loss=reconstruct_loss.cuda()
clf_loss=clf_loss.cuda()

# Optmizer
opt_vae = optim.Adam(list(vae.parameters()), lr=vae_learning_rate, betas=vae_betas)
# opt_df = optim.Adam(list(d_feat.parameters()), lr=df_learning_rate, betas=df_betas)# Dv
opt_dp = optim.Adam(list(d_pix.parameters()), lr=dp_learning_rate, betas=dp_betas)# Dx

# training
vae,train()
d_pix.train()

# Domain code setting
domain_code = np.concatenate([np.repeat(np.array([[*([1]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3))]]),batch_size,axis=0),
                              np.repeat(np.array([[*([0]*int(code_dim/3)),
                                                   *([1]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3))]]),batch_size,axis=0),
                              np.repeat(np.array([[*([0]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3)),
                                                   *([1]*int(code_dim/3))]]),batch_size,axis=0)],
                              axis=0)

domain_code = torch.FloatTensor(domain_code)

### Messy, torch.randperm will be better approach
# forword translation code : A->B->C->A
forword_code = np.concatenate([np.repeat(np.array([[*([0]*int(code_dim/3)),
                                                   *([1]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3))]]),batch_size,axis=0),
                              np.repeat(np.array([[*([0]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3)),
                                                   *([1]*int(code_dim/3))]]),batch_size,axis=0),
                              np.repeat(np.array([[*([1]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3))]]),batch_size,axis=0)],
                              axis=0)

forword_code = torch.FloatTensor(forword_code)

# backword translation code : C->B->A->C
backword_code = np.concatenate([np.repeat(np.array([[*([0]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3)),
                                                   *([1]*int(code_dim/3))]]),batch_size,axis=0),
                              np.repeat(np.array([[*([1]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3))]]),batch_size,axis=0),
                              np.repeat(np.array([[*([0]*int(code_dim/3)),
                                                   *([1]*int(code_dim/3)),
                                                   *([0]*int(code_dim/3))]]),batch_size,axis=0)],
                              axis=0)

backword_code = torch.FloatTensor(backword_code)
