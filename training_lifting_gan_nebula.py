#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys
import numpy as np
from nets.PosePriorNetwork_nebula import PosePriorNetwork
#from data.BinaryDbReaderSTB import BinaryDbReaderSTB
from data.BinaryDbReader_gan import BinaryDbReader
from utils.general import LearningRateScheduler, load_weights_from_snapshot
import matplotlib.pyplot as plt
from pca import pca
# Chose which variant to evaluate
# VARIANT = 'direct'
# VARIANT = 'bottleneck'
# VARIANT = 'local'
# VARIANT = 'local_w_xyz_loss'
VARIANT = 'proposed'
PATH_TO_LIFTING_SNAPSHOTS = './snapshots_lifting_gan_nebula_7256_z/' #ckp = 125k
# training parameters
train_para = {'lr': [1e-5, 1e-6],
              'lr_iter': [60000],
              'max_iter': 150000,
              'show_loss_freq': 1000,
              'snapshot_freq': 5000,
              'snapshot_dir': './snapshots_lifting_gan_nebula_7256_z+0d/'}#snapshots_lifting_gan_nebula_10256_z+0d is the best model in rebuttal phase
# get dataset
dataset = BinaryDbReader(mode='training_gan',
                         batch_size=16, shuffle=True, hand_crop=False, use_wrist_coord=True,
                         coord_uv_noise=True, crop_center_noise=False, crop_offset_noise=False, crop_scale_noise = False)
#dataset = BinaryDbReaderSTB(mode='training_stb',
                         #batch_size=16, shuffle=True, hand_crop=True, use_wrist_coord=True,
                         #coord_uv_noise=True, crop_center_noise=False, crop_offset_noise=False, crop_scale_noise = False)
# build network graph
data = dataset.get()

# build network
net = PosePriorNetwork(VARIANT)

# feed trough network
evaluation = tf.placeholder_with_default(True, shape=())
coord3d_list = net.inference(data['scoremap'], data['hand_side'], evaluation)
coord_xyz_rel_normed = coord3d_list['coord_xyz_rel_normed']
coord3d_pred = coord3d_list['coord3d']
R = coord3d_list['R']
loss_0d = coord3d_list['loss_0d']
# Start TF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
loss_org = 0.0

loss_org += tf.reduce_mean(tf.square(coord3d_pred - data['keypoint_xyz21_can']))
loss_org += tf.reduce_mean(tf.square(R - data['rot_mat']))

loss = loss_org + 0.01*loss_0d + tf.reduce_mean(coord3d_list['loss_z']) #+tf.reduce_mean(coord3d_list['nebula_z'])
#loss = loss_org + tf.reduce_mean(coord3d_list['loss_z'])
# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4.0)
last_cpt = tf.train.latest_checkpoint(PATH_TO_LIFTING_SNAPSHOTS)
load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):

    #_, loss_v, org, xyz_can, rot, xyz_normed = sess.run([train_op, loss, loss_org, data['keypoint_xyz21_can'], data['rot_mat'], data['keypoint_xyz21_normed']])

    _, loss_v, org, d0 = sess.run([train_op, loss, loss_org, loss_0d])
    nebula3d = sess.run([coord3d_list['nebula3d']])

    #import ipdb;ipdb.set_trace()
    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e,  Loss_0d %.4e, Loss_org %.4e' % (i, loss_v, d0, org))#,'lossd: ',d0,', ', d1)#,', ', d2,', ', d3)
        #label_viz = []
        nebula_index = []
        gt = []
        latent = []
        coord_can63 = []
        for m in range(0,40):
            viz_data = sess.run(data)
            coord_can63  = np.append(coord_can63, sess.run([coord3d_list['coord_can63']], feed_dict={coord3d_list['scoremap']: viz_data['scoremap'],coord3d_list['hand_side']: viz_data['hand_side']})[0    ])
            latent = np.append(latent, sess.run([coord3d_list['x_label']], feed_dict={coord3d_list['scoremap']: viz_data['scoremap'],coord3d_list['hand_side']: viz_data['hand_side']})[0])             
            #batch_label  = viz_data['label']
            #label_viz = np.append(label_viz, batch_label).astype(int)
            nebula_index = np.append(nebula_index, sess.run([coord3d_list['index']], feed_dict={coord3d_list['scoremap']: viz_data['scoremap'],coord3d_list['hand_side']: viz_data['hand_side']})[0])
            batch_gt = np.reshape(viz_data['keypoint_xyz21_can'],(-1,63))
            gt = np.append(gt, batch_gt).astype(int)
            nebula3d = np.squeeze(nebula3d)
        latent = np.reshape(latent, (-1, 256))
        coord_can63 = np.reshape(coord_can63, (-1, 63))
        gt = np.reshape(gt, (-1, 63))
        #feat_viz, V = pca(z_viz, dim_remain=2)
        _, V = pca(latent, dim_remain=2)
        _, V_can = pca(coord_can63, dim_remain=2)
        latent_viz = np.matmul(latent, V)
        nebula_viz = np.matmul(nebula3d, V)
        coord63_viz = np.matmul(coord_can63, V_can)
        gt_viz = np.matmul(gt, V_can)
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.set_axis_off()
        ax.scatter(coord63_viz[:, 0], coord63_viz[:, 1], c=nebula_index, alpha=0.3, cmap='Accent')
        fig1.savefig(train_para['snapshot_dir']+'/pca_pred_can.png', transparent=True)
        plt.close(fig1)
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.set_axis_off()
        ax.scatter(gt_viz[:, 0], gt_viz[:, 1], c=nebula_index, alpha=0.3, cmap='Accent')
        fig2.savefig(train_para['snapshot_dir']+'/pca_gt_can.png', transparent=True)
        plt.close(fig2)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.scatter(latent_viz[:, 0], latent_viz[:, 1], c=nebula_index, alpha=0.3, s=150, cmap='Accent')
        ax.scatter(nebula_viz[:, 0], nebula_viz[:, 1], marker='H', alpha=0.8, s=150, cmap='black')
        fig.savefig(train_para['snapshot_dir']+'/scatter.png', transparent=True)
        plt.close(fig)
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
