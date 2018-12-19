import yaml
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
from torchvision.utils import save_image
from tensorboardX import SummaryWriter


class Solver(object):
    def __init__(self, conf):
        self.exp_name = conf['exp_setting']['exp_name']  # each dir name
        self.img_size = conf['exp_setting']['img_size']
        self.resize_size = conf['exp_setting']['resize_size']
        self.img_depth = conf['exp_setting']['img_depth']
        self.data_root = conf['exp_setting']['data_root']
        self.batch_size = conf['trainer']['batch_size']
        self.trainer_conf = conf['trainer']
        # Load dataset
        self.domain_list = conf['exp_setting']['domain']
        self.domain_size = len(self.domain_list)
        # reload model
        self.reload_model = conf['exp_setting']['reload_model']
        self.idx = conf['exp_setting']['reload_idx']
        if self.trainer_conf['save_checkpoint']:
            self.model_path_vae = conf['exp_setting']['checkpoint_dir'] + self.exp_name + '/vae/'
            self.model_path_dx = conf['exp_setting']['checkpoint_dir'] + self.exp_name + '/dx/'
            self.model_path_dv = conf['exp_setting']['checkpoint_dir'] + self.exp_name + '/dv/'
            self.makedir(self.model_path_vae)
            self.makedir(self.model_path_dx)
            self.makedir(self.model_path_dv)

        if self.trainer_conf['save_log'] or self.trainer_conf['save_fig']:  # for tensorboard
            if os.path.exists(conf['exp_setting']['log_dir'] + self.exp_name):
                shutil.rmtree(conf['exp_setting']['log_dir'] + self.exp_name)
            self.writer = SummaryWriter(conf['exp_setting']['log_dir'] + self.exp_name)

        # Fix seed
        np.random.seed(conf['exp_setting']['seed'])
        _ = torch.manual_seed(conf['exp_setting']['seed'])

        self.build_model()

    def build_model(self):
        # Load Model
        self.enc_dim = conf['model']['vae']['encoder'][-1][1]
        self.code_dim = conf['model']['vae']['code_dim']
        self.vae_learning_rate = conf['model']['vae']['lr']
        self.vae_betas = tuple(conf['model']['vae']['betas'])
        self.df_learning_rate = conf['model']['D_feat']['lr']
        self.df_betas = tuple(conf['model']['D_feat']['betas'])
        self.update_Dv = conf['model']['D_feat']['update_Dv']
        self.dp_learning_rate = conf['model']['D_pix']['lr']
        self.dp_betas = tuple(conf['model']['D_pix']['betas'])

        self.vae = LoadModel('vae', conf['model']['vae'], self.resize_size, self.img_depth)
        self.d_feat = LoadModel('nn', conf['model']['D_feat'], self.resize_size, self.enc_dim)
        self.d_pix = LoadModel('cnn', conf['model']['D_pix'], self.resize_size, self.img_depth)

        if self.reload_model:
            if self.model_path_vae != '':
                self.vae = torch.load(self.model_path_vae + '{}'.format(str(self.idx)) + 'vae.pkl')
            if self.model_path_dv != '':
                self.d_feat = torch.load(self.model_path_dv + '{}'.format(str(self.idx)) + 'dv.pkl')
            if self.model_path_dx != '':
                self.d_pix = torch.load(self.model_path_dx + '{}'.format(str(self.idx)) + 'dx.pkl')
            print('Reload model ' + str(self.idx))
        self.reconstruct_loss = torch.nn.MSELoss()
        self.clf_loss = nn.BCEWithLogitsLoss()

        # Use cuda
        self.vae = self.vae.cuda()
        self.d_feat = self.d_feat.cuda()
        self.d_pix = self.d_pix.cuda()
        self.reconstruct_loss = self.reconstruct_loss.cuda()
        self.clf_loss = self.clf_loss.cuda()

        # Optmizer
        self.opt_vae = optim.Adam(list(self.vae.parameters()), lr=self.vae_learning_rate, betas=self.vae_betas)
        self.opt_df = optim.Adam(list(self.d_feat.parameters()), lr=self.df_learning_rate, betas=self.df_betas)  # Dv
        self.opt_dp = optim.Adam(list(self.d_pix.parameters()), lr=self.dp_learning_rate, betas=self.dp_betas)  # Dx

    def train(self):
        domain_code = np.concatenate([np.repeat(np.array([np.eye(self.domain_size)[0]]), self.batch_size, axis=0),
                                      np.repeat(np.array([np.eye(self.domain_size)[1]]), self.batch_size, axis=0),
                                      np.repeat(np.array([np.eye(self.domain_size)[2]]), self.batch_size, axis=0)],
                                     axis=0)
        domain_code = torch.FloatTensor(domain_code)

        ### Messy, torch.randperm will be better approach
        # forword translation code : A->B->C->A
        forword_code = np.concatenate([np.repeat(np.array([np.eye(self.domain_size)[1]]), self.batch_size, axis=0),
                                       np.repeat(np.array([np.eye(self.domain_size)[2]]), self.batch_size, axis=0),
                                       np.repeat(np.array([np.eye(self.domain_size)[0]]), self.batch_size, axis=0)],
                                      axis=0)

        forword_code = torch.FloatTensor(forword_code)

        # backword translation code : C->B->A->C
        backword_code = np.concatenate([np.repeat(np.array([np.eye(self.domain_size)[2]]), self.batch_size, axis=0),
                                        np.repeat(np.array([np.eye(self.domain_size)[0]]), self.batch_size, axis=0),
                                        np.repeat(np.array([np.eye(self.domain_size)[1]]), self.batch_size, axis=0)],
                                       axis=0)

        backword_code = torch.FloatTensor(backword_code)

        # Loss weight setting
        loss_lambda = {}
        for k in self.trainer_conf['lambda'].keys():
            init = self.trainer_conf['lambda'][k]['init']
            final = self.trainer_conf['lambda'][k]['final']
            step = self.trainer_conf['lambda'][k]['step']
            loss_lambda[k] = {}
            loss_lambda[k]['cur'] = init
            loss_lambda[k]['inc'] = (final - init) / step
            loss_lambda[k]['final'] = final

        # Training
        global_step = 0
        if self.reload_model:
            global_step = self.idx
        print(global_step)
        self.vae.train()
        self.d_feat.train()
        self.d_pix.train()

        while global_step < self.trainer_conf['total_step']:
            train_loader = self.makeloader('train', self.batch_size)
            for (a, b, c) in zip(train_loader[0], train_loader[1], train_loader[2]):
                # data augmentation
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

                # Train Feature Discriminator
                self.opt_df.zero_grad()

                enc_x = self.vae(input_img, return_enc=True).detach()
                code_pred = self.d_feat(enc_x)

                df_loss = self.clf_loss(code_pred, code)
                df_loss.backward()

                self.opt_df.step()

                # Train Pixel Discriminator
                self.opt_dp.zero_grad()

                pix_real_pred, pix_real_code_pred = self.d_pix(input_img)

                fake_img = self.vae(input_img, insert_attrs=trans_code)[0].detach()
                pix_fake_pred, _ = self.d_pix(fake_img)

                pix_real_pred = pix_real_pred.mean()
                pix_fake_pred = pix_fake_pred.mean()

                gp = loss_lambda['gp']['cur'] * calc_gradient_penalty(self.d_pix, input_img.data, fake_img.data)
                pix_code_loss = self.clf_loss(pix_real_code_pred, code)

                d_pix_loss = pix_code_loss + pix_fake_pred - pix_real_pred + gp
                d_pix_loss.backward()

                self.opt_dp.step()

                # Train VAE
                self.opt_vae.zero_grad()

                ### Reconstruction Phase
                recon_batch, mu, logvar = self.vae(input_img, insert_attrs=code)
                mse, kl = vae_loss(recon_batch, input_img, mu, logvar, self.reconstruct_loss)  # .view(batch_size,-1)
                recon_loss = (loss_lambda['pix_recon']['cur'] * mse + loss_lambda['kl']['cur'] * kl)
                recon_loss.backward()

                ### Feature space adversarial Phase
                enc_x = self.vae(input_img, return_enc=True)
                domain_pred = self.d_feat(enc_x)
                adv_code_loss = self.clf_loss(domain_pred, invert_code)

                feature_loss = loss_lambda['feat_domain']['cur'] * adv_code_loss
                feature_loss.backward()

                ### Pixel space adversarial Phase
                enc_x = self.vae(input_img, return_enc=True).detach()
                fake_img = self.vae.decode(enc_x, trans_code)
                recon_enc_x = self.vae(fake_img, return_enc=True)
                adv_pix_loss, pix_code_pred = self.d_pix(fake_img)
                adv_pix_loss = adv_pix_loss.mean()
                pix_clf_loss = self.clf_loss(pix_code_pred, trans_code)

                pixel_loss = - loss_lambda['pix_adv']['cur'] * adv_pix_loss + loss_lambda['pix_clf'][
                                                                                  'cur'] * pix_clf_loss
                pixel_loss.backward()

                self.opt_vae.step()

                # End of step
                print('Step', global_step, end='\r', flush=True)
                global_step += 1

                # Records
                if self.trainer_conf['save_log'] and (global_step % self.trainer_conf['verbose_step'] == 0):
                    self.writer.add_scalar('MSE', mse.data[0], global_step)
                    self.writer.add_scalar('KL', kl.data[0], global_step)
                    self.writer.add_scalar('gp', gp.data[0], global_step)
                    self.writer.add_scalars('Pixel_Distance', {'real': pix_real_pred.data[0],
                                                               'fake': pix_fake_pred.data[0]}, global_step)
                    self.writer.add_scalars('Code_loss', {
                        # 'feature': df_loss.data[0],
                        'pixel': pix_code_loss.data[0],
                        # 'adv_feature': feature_loss.data[0],
                        'adv_pixel': pix_clf_loss.data[0]}, global_step)

                    self.writer.add_scalars('Code_loss', {
                        'feature': df_loss.data[0],
                        'pixel': pix_code_loss.data[0],
                        'adv_feature': feature_loss.data[0],
                        'adv_pixel': pix_clf_loss.data[0]}, global_step)

                # update lambda
                for k in loss_lambda.keys():
                    if loss_lambda[k]['inc'] * loss_lambda[k]['cur'] < loss_lambda[k]['inc'] * loss_lambda[k]['final']:
                        loss_lambda[k]['cur'] += loss_lambda[k]['inc']

                if global_step % self.trainer_conf['checkpoint_step'] == 0 and self.trainer_conf[
                    'save_checkpoint'] and not \
                        self.trainer_conf['save_best_only']:
                    torch.save(self.vae, self.model_path_vae + '{}'.format(global_step) + 'vae.pkl')
                    torch.save(self.d_pix, self.model_path_dx + '{}'.format(global_step) + 'dx.pkl')
                    torch.save(self.d_feat, self.model_path_dv + '{}'.format(global_step) + 'dv.pkl')

                ### Show result
                if global_step % self.trainer_conf['plot_step'] == 0:
                    # if global_step%5==0:
                    self.vae.eval()

                    # test data
                    test_loader = self.makeloader('test', 1)
                    for d1, d2, d3 in zip(test_loader[0], test_loader[1],
                                          test_loader[2]):  # from the same people or not
                        a_test_sample = d1[0].type(torch.FloatTensor)
                        b_test_sample = d2[0].type(torch.FloatTensor)
                        c_test_sample = d3[0].type(torch.FloatTensor)
                    # Reconstruct
                    tmp = interpolate_vae_3d(self.vae, a_test_sample, b_test_sample, c_test_sample, attr_max=1.0,
                                             attr_dim=self.code_dim)
                    fig1 = (tmp + 1) / 2

                    # Generate
                    tmp = interpolate_vae_3d(self.vae, a_test_sample, b_test_sample, c_test_sample, attr_max=1.0,
                                             random_test=True,
                                             sd=conf['exp_setting']['seed'], attr_dim=self.code_dim)
                    fig2 = (tmp + 1) / 2

                    if self.trainer_conf['save_fig']:
                        self.writer.add_image('interpolate', torch.FloatTensor(np.transpose(fig1, (2, 0, 1))),
                                              global_step)
                        self.writer.add_image('random generate', torch.FloatTensor(np.transpose(fig2, (2, 0, 1))),
                                              global_step)

                    self.vae.train()

    def test(self, idx):
        test_path = './test/' + self.exp_name
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        vae = torch.load(self.model_path_vae + '{}'.format(str(idx)) + 'vae.pkl')
        vae.eval()

        # test data
        test_loader = self.makeloader('test', 1)
        for d1, d2, d3 in zip(test_loader[0], test_loader[1], test_loader[2]):  # from the same people or not
            a_test_sample = d1[0].type(torch.FloatTensor)
            b_test_sample = d2[0].type(torch.FloatTensor)
            c_test_sample = d3[0].type(torch.FloatTensor)

        # Reconstruct
        tmp = interpolate_vae_3d(vae, a_test_sample, b_test_sample, c_test_sample, attr_max=1.0,
                                 attr_dim=self.code_dim)
        fig1 = (tmp + 1) / 2

        # Generate
        tmp = interpolate_vae_3d(vae, a_test_sample, b_test_sample, c_test_sample, attr_max=1.0, random_test=True,
                                 sd=conf['exp_setting']['seed'], attr_dim=self.code_dim)
        fig2 = (tmp + 1) / 2
        save_image(torch.FloatTensor(np.transpose(fig1, (2, 0, 1))), os.path.join(test_path, str(idx) + 'g1.jpg'),
                   nrow=9, padding=0)
        save_image(torch.FloatTensor(np.transpose(fig2, (2, 0, 1))), os.path.join(test_path, str(idx) + 'g2.jpg'),
                   nrow=9, padding=0)

        # plt.imsave(os.path.join(test_path, 'g1.jpg'), np.transpose(fig1, (0,1,2)))
        # plt.imsave(os.path.join(test_path, 'g2.jpg'), np.transpose(fig2, (1,2, 0)))

    def makeloader(self, type, size):  # type='train'/'test',size=pic numbers:bz/1

        data_loader = []
        # self.test_loader = []

        for i, item in enumerate(self.domain_list):
            traindata = LoadDataset('face', self.data_root, size, type, resize_size=self.resize_size, style=item)
            data_loader.append(traindata)
        return data_loader

    def makedir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)


if __name__ == '__main__':
    cudnn.benchmark = True
    config_path = sys.argv[1]
    # './config/oriUFDN.yaml'
    './config/La(onlychange256).yaml'
    conf = yaml.load(open(config_path, 'r'))
    train_flag = conf['exp_setting']['train_flag']
    test_flag = conf['exp_setting']['test_flag']
    test_idx = conf['exp_setting']['test_idx']

    solver = Solver(conf)
    if train_flag:
        solver.train()
    if test_flag:
        solver.test(idx=test_idx)
