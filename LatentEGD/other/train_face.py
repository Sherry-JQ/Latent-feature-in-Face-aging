import yaml
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
import shutil
import numpy as np
import torch
from torch.backends import cudnn
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable, grad

from src.data import LoadDataset
from src.ufdn import LoadModel

from src.util import vae_loss, calc_gradient_penalty, interpolate_vae_3d

from tensorboardX import SummaryWriter

#########################################################
# Experiment Setting
# if __name__ == '__main__':
cudnn.benchmark = True
config_path = './config/oriUFDN.yaml'
    # sys.argv[1]
# './config/oriUFDN.yaml'
conf = yaml.load(open(config_path, 'r'))
exp_name = conf['exp_setting']['exp_name']  # each dir name
img_size = conf['exp_setting']['img_size']
resize_size = conf['exp_setting']['resize_size']
img_depth = conf['exp_setting']['img_depth']
trainer_conf = conf['trainer']

if trainer_conf['save_checkpoint']:
    model_path_vae = conf['exp_setting']['checkpoint_dir'] + exp_name + '/vae/'
    model_path_dx = conf['exp_setting']['checkpoint_dir'] + exp_name + '/dx/'
    model_path_dv = conf['exp_setting']['checkpoint_dir'] + exp_name + '/dv/'

    if not os.path.exists(model_path_vae):
        os.makedirs(model_path_vae)
    if not os.path.exists(model_path_dx):
        os.makedirs(model_path_dx)
    if not os.path.exists(model_path_dv):
        os.makedirs(model_path_dv)
    # model_path_vae = model_path_vae + '{}'
    # model_path_dx = model_path_dx + '{}'
    # model_path_dv = model_path_dv + '{}'


if trainer_conf['save_log'] or trainer_conf['save_fig']:
    if os.path.exists(conf['exp_setting']['log_dir'] + exp_name):
        shutil.rmtree(conf['exp_setting']['log_dir'] + exp_name)
    writer = SummaryWriter(conf['exp_setting']['log_dir'] + exp_name)
# Fix seed
np.random.seed(conf['exp_setting']['seed'])
_ = torch.manual_seed(conf['exp_setting']['seed'])

data_root = conf['exp_setting']['data_root']
batch_size = conf['trainer']['batch_size']
#####################################################################

# Load dataset
domain_list = conf['exp_setting']['domain']
domain_size = len(domain_list)

train_loader = []
test_loader = []
traindata_size = 10000
for i, item in enumerate(domain_list):
    # setattr('domain_',str(i), item)
    traindata = LoadDataset('face', data_root, batch_size, 'train', resize_size=resize_size, style=item)
    traindata_size = min(traindata_size, len(traindata))
    train_loader.append(traindata)
    testdata = LoadDataset('face', data_root, 1, 'test', resize_size=resize_size, style=item)
    test_loader.append(testdata)

# # a_loader =
# # # print (a_loader)
# # b_loader = LoadDataset('face', data_root, batch_size, 'train', style=doamin_b)
# # c_loader = LoadDataset('face', data_root, batch_size, 'train', style=doamin_c)
#
# a_test = LoadDataset('face', data_root, 1, 'test', style=domain_a)
# b_test = LoadDataset('face', data_root, 1, 'test', style=doamin_b)
# c_test = LoadDataset('face', data_root, 1, 'test', style=doamin_c)
#
for d1, d2, d3 in zip(test_loader[0], test_loader[1], test_loader[2]):  # from the same people or not
    a_test_sample = d1[0].type(torch.FloatTensor)
    b_test_sample = d2[0].type(torch.FloatTensor)
    c_test_sample = d3[0].type(torch.FloatTensor)
    break

# Load Model
enc_dim = conf['model']['vae']['encoder'][-1][1]
code_dim = conf['model']['vae']['code_dim']
vae_learning_rate = conf['model']['vae']['lr']
vae_betas = tuple(conf['model']['vae']['betas'])
df_learning_rate = conf['model']['D_feat']['lr']
df_betas = tuple(conf['model']['D_feat']['betas'])
update_Dv = conf['model']['D_feat']['update_Dv']
dp_learning_rate = conf['model']['D_pix']['lr']
dp_betas = tuple(conf['model']['D_pix']['betas'])


vae = LoadModel('vae', conf['model']['vae'], resize_size, img_depth)
d_feat = LoadModel('nn', conf['model']['D_feat'], resize_size, enc_dim)
d_pix = LoadModel('cnn', conf['model']['D_pix'], resize_size, img_depth)
# if model_path_vae != '':
#     vae.load_state_dict(torch.load(model_path_vae))
# if model_path_dx != '':
#     d_pix.load_state_dict(torch.load(model_path_dx))
# if model_path_dv != '':
#     d_feat.load_state_dict(torch.load(model_path_dv))

reconstruct_loss = torch.nn.MSELoss()
clf_loss = nn.BCEWithLogitsLoss()

# Use cuda
vae = vae.cuda()
d_feat = d_feat.cuda()
d_pix = d_pix.cuda()
reconstruct_loss = reconstruct_loss.cuda()
clf_loss = clf_loss.cuda()

# Optmizer
opt_vae = optim.Adam(list(vae.parameters()), lr=vae_learning_rate, betas=vae_betas)
opt_df = optim.Adam(list(d_feat.parameters()), lr=df_learning_rate, betas=df_betas)  # Dv
opt_dp = optim.Adam(list(d_pix.parameters()), lr=dp_learning_rate, betas=dp_betas)  # Dx




# Domain code setting

domain_code = np.concatenate([np.repeat(np.array([np.eye(domain_size)[0]]), batch_size, axis=0),
                              np.repeat(np.array([np.eye(domain_size)[1]]), batch_size, axis=0),
                              np.repeat(np.array([np.eye(domain_size)[2]]), batch_size, axis=0)], axis=0)
domain_code = torch.FloatTensor(domain_code)

### Messy, torch.randperm will be better approach
# forword translation code : A->B->C->A
forword_code = np.concatenate([np.repeat(np.array([np.eye(domain_size)[1]]), batch_size, axis=0),
                               np.repeat(np.array([np.eye(domain_size)[2]]), batch_size, axis=0),
                               np.repeat(np.array([np.eye(domain_size)[0]]), batch_size, axis=0)], axis=0)

forword_code = torch.FloatTensor(forword_code)

# backword translation code : C->B->A->C
backword_code = np.concatenate([np.repeat(np.array([np.eye(domain_size)[2]]), batch_size, axis=0),
                                np.repeat(np.array([np.eye(domain_size)[0]]), batch_size, axis=0),
                                np.repeat(np.array([np.eye(domain_size)[1]]), batch_size, axis=0)], axis=0)

backword_code = torch.FloatTensor(backword_code)

# Loss weight setting
loss_lambda = {}
for k in trainer_conf['lambda'].keys():
    init = trainer_conf['lambda'][k]['init']
    final = trainer_conf['lambda'][k]['final']
    step = trainer_conf['lambda'][k]['step']
    loss_lambda[k] = {}
    loss_lambda[k]['cur'] = init
    loss_lambda[k]['inc'] = (final - init) / step
    loss_lambda[k]['final'] = final

# Training
global_step = 0
print(global_step)

train_flag=False
# Training

vae.train()
d_feat.train()
d_pix.train()
if train_flag:
    while global_step < trainer_conf['total_step']:

        for (a, b, c) in zip(train_loader[0], train_loader[1], train_loader[2]):
            # data augmentation
            #     input_img=None
            input_img = torch.cat([a.type(torch.FloatTensor),
                               b.type(torch.FloatTensor),
                               c.type(torch.FloatTensor)], dim=0)
            input_img = Variable(input_img.cuda(), requires_grad=False)

            code = Variable(torch.FloatTensor(domain_code).cuda(), requires_grad=False)
            invert_code = 1 - code

            if global_step % 2 == 0:
                trans_code = Variable(torch.FloatTensor(forword_code).cuda(), requires_grad=False)
            else:
                trans_code = Variable(torch.FloatTensor(backword_code).cuda(), requires_grad=False)

            if update_Dv:
                # Train Feature Discriminator
                opt_df.zero_grad()

                enc_x = vae(input_img, return_enc=True).detach()
                code_pred = d_feat(enc_x)

                df_loss = clf_loss(code_pred, code)
                df_loss.backward()

                opt_df.step()

            # Train Pixel Discriminator
            opt_dp.zero_grad()

            pix_real_pred, pix_real_code_pred = d_pix(input_img)

            fake_img = vae(input_img, insert_attrs=trans_code)[0].detach()
            pix_fake_pred, _ = d_pix(fake_img)

            pix_real_pred = pix_real_pred.mean()
            pix_fake_pred = pix_fake_pred.mean()

            gp = loss_lambda['gp']['cur'] * calc_gradient_penalty(d_pix, input_img.data, fake_img.data)
            pix_code_loss = clf_loss(pix_real_code_pred, code)

            d_pix_loss = pix_code_loss + pix_fake_pred - pix_real_pred + gp
            d_pix_loss.backward()

            opt_dp.step()

            # Train VAE
            opt_vae.zero_grad()

            ### Reconstruction Phase
            recon_batch, mu, logvar = vae(input_img, insert_attrs=code)
            mse, kl = vae_loss(recon_batch, input_img, mu, logvar, reconstruct_loss)  # .view(batch_size,-1)
            recon_loss = (loss_lambda['pix_recon']['cur'] * mse + loss_lambda['kl']['cur'] * kl)
            recon_loss.backward()

            if update_Dv:
                ### Feature space adversarial Phase
                enc_x = vae(input_img, return_enc=True)
                domain_pred = d_feat(enc_x)
                adv_code_loss = clf_loss(domain_pred, invert_code)

                feature_loss = loss_lambda['feat_domain']['cur'] * adv_code_loss
                feature_loss.backward()

            ### Pixel space adversarial Phase
            enc_x = vae(input_img, return_enc=True).detach()

            fake_img = vae.decode(enc_x, trans_code)
            recon_enc_x = vae(fake_img, return_enc=True)
            adv_pix_loss, pix_code_pred = d_pix(fake_img)
            adv_pix_loss = adv_pix_loss.mean()
            pix_clf_loss = clf_loss(pix_code_pred, trans_code)

            pixel_loss = - loss_lambda['pix_adv']['cur'] * adv_pix_loss + loss_lambda['pix_clf']['cur'] * pix_clf_loss
            pixel_loss.backward()

            opt_vae.step()

            # End of step
            print('Step', global_step, end='\r', flush=True)
            global_step += 1

            # Records
            if trainer_conf['save_log'] and (global_step % trainer_conf['verbose_step'] == 0):
                writer.add_scalar('MSE', mse.data[0], global_step)
                writer.add_scalar('KL', kl.data[0], global_step)
                writer.add_scalar('gp', gp.data[0], global_step)
                writer.add_scalars('Pixel_Distance', {'real': pix_real_pred.data[0],
                                                      'fake': pix_fake_pred.data[0]}, global_step)
                writer.add_scalars('Code_loss', {
                    # 'feature': df_loss.data[0],
                    'pixel': pix_code_loss.data[0],
                    # 'adv_feature': feature_loss.data[0],
                    'adv_pixel': pix_clf_loss.data[0]}, global_step)
                if update_Dv:
                    writer.add_scalars('Code_loss', {
                    'feature':df_loss.data[0],
                    'pixel': pix_code_loss.data[0],
                    'adv_feature':feature_loss.data[0],
                    'adv_pixel': pix_clf_loss.data[0]}, global_step)

            # update lambda
            for k in loss_lambda.keys():
                if loss_lambda[k]['inc'] * loss_lambda[k]['cur'] < loss_lambda[k]['inc'] * loss_lambda[k]['final']:
                    loss_lambda[k]['cur'] += loss_lambda[k]['inc']

            if global_step % trainer_conf['checkpoint_step'] == 0 and trainer_conf['save_checkpoint'] and not trainer_conf[
                'save_best_only']:
                torch.save(vae, model_path_vae+'{}'.format(global_step) + '.vae')
                torch.save(d_pix, model_path_dx+'{}'.format(global_step) + '.dx')
                torch.save(d_feat, model_path_vae+'{}'.format(global_step) + '.dv')



            ### Show result
            if global_step % trainer_conf['plot_step'] == 0:
                # if global_step%5==0:
                vae.eval()

                # Reconstruct
                tmp = interpolate_vae_3d(vae, a_test_sample, b_test_sample, c_test_sample, attr_max=1.0, attr_dim=code_dim)
                fig1 = (tmp + 1) / 2

                # Generate
                tmp = interpolate_vae_3d(vae, a_test_sample, b_test_sample, c_test_sample, attr_max=1.0, random_test=True,
                                         sd=conf['exp_setting']['seed'], attr_dim=code_dim)
                fig2 = (tmp + 1) / 2

                if trainer_conf['save_fig']:
                    writer.add_image('interpolate', torch.FloatTensor(np.transpose(fig1, (2, 0, 1))), global_step)
                    writer.add_image('random generate', torch.FloatTensor(np.transpose(fig2, (2, 0, 1))), global_step)

                vae.train()

# testing
test_flag=True
if test_flag:
    test_path = 'test/' + exp_name + '/'
    if not os.path.exists(test_path):
            os.makedirs(test_path)
    torch.load(torch.load(model_path_vae+'/99500.vae'))
    vae.eval()

    # Reconstruct
    tmp = interpolate_vae_3d(vae, a_test_sample, b_test_sample, c_test_sample, attr_max=1.0, attr_dim=code_dim)
    fig1 = (tmp + 1) / 2

    # Generate
    tmp = interpolate_vae_3d(vae, a_test_sample, b_test_sample, c_test_sample, attr_max=1.0, random_test=True,
                             sd=conf['exp_setting']['seed'], attr_dim=code_dim)
    fig2 = (tmp + 1) / 2
    plt.imsave(os.path.join(test_path,'g1.jpg'), np.transpose(fig1, (2, 0, 1)))
