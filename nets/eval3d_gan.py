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
""" Script for evaluation of different Lifting variants on full scale images.

    This allows to reproduce Table 2 of the paper R-val "Average median error":
    Method      | Number in the paper                   | Our result with TF 1.3

    Direct      | 20.9                                  | 20.848 mm
    Bottleneck  | Number in the paper is *not* correct  | 21.907 mm
    Local       | 39.1                                  | 39.121 mm
    Proposed    | 18.8                                  | 18.840 mm


    Also there is one new variant that was not included in the paper as it is more current work.
    Its the like local, but with the loss in xyz coordinate frame, which seems to work better:
    Local with XYZ Loss  21.950 mm
"""""
from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.BinaryDbReader_gan import *
#from data.BinaryDbReaderSTB import BinaryDbReaderSTB
#from nets.PosePriorNetwork_mlvae_2_noflip import PosePriorNetwork
#from nets.PosePriorNetwork_mlold_noflip import PosePriorNetwork
#from nets.PosePriorNetwork_mlvae import PosePriorNetwork
#from nets.PosePriorNetwork_un3 import PosePriorNetwork
#from nets.PosePriorNetwork_gan_un0221 import PosePriorNetwork
#from nets.PosePriorNetwork import PosePriorNetwork
#from nets.PosePriorNetwork_gan import PosePriorNetwork
#from nets.PosePriorNetwork_mlold_gan import PosePriorNetwork
from nets.PosePriorNetwork_nebula import PosePriorNetwork
#from nets.PosePriorNetwork_gan_un0221 import PosePriorNetwork
from utils.general_2 import plot_hand, plot_hand_3d, get_stb_ref_curves, EvalUtil, load_weights_from_snapshot

# Chose which variant to evaluate
USE_RETRAINED = True
# VARIANT = 'direct'
# VARIANT = 'bottleneck'
# VARIANT = 'local'
# VARIANT = 'local_w_xyz_loss'
VARIANT = 'proposed'

# get dataset
#dataset = BinaryDbReaderSTB(mode='evaluation_stb',  shuffle=False, hand_crop=True, use_wrist_coord=True)
dataset = BinaryDbReader(mode='evaluation_gan',  shuffle=False, hand_crop=False, use_wrist_coord=True)
# build network graph
data = dataset.get()

# build network
net = PosePriorNetwork(VARIANT)

# feed through network
evaluation = tf.placeholder_with_default(True, shape=())
#pred3d_list = net.inference(data['scoremap'], data['hand_side'], None, evaluation)
pred3d_list = net.inference(data['scoremap'], data['hand_side'], evaluation)
coord3d_pred = pred3d_list['coord_xyz_rel_normed']
coord3d_can = pred3d_list['coord3d']
coord3d_gt = data['keypoint_xyz21']
R = pred3d_list['R']
rot_gt = data['rot_mat']
#coord3d_gt = data['keypoint_xyz21_can']
# Start TF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.train.start_queue_runners(sess=sess)

# initialize network with weights used in the paper
if USE_RETRAINED:

    # retrained version: HandSegNet
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_stb_org/')
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_gan_un_wolossz100/')
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_gan_normed/')
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_mlold_gan/')
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_STB_un0221_org+0d/')
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_gan_nebula_10256_z/')
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_gan_nebula_10256_z+0d/')
    last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_gan_nebula_7256_z+0d/')
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_stb_nebula_10256_z2/')
    #last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_stb_nebula_10256_z2+0d/984/')

    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
else:
    net.init(sess, weight_files=['./weights/lifting-%s.pickle' % VARIANT])

util = EvalUtil()
# iterate dataset
'''
#stb, rhd& hop
for i in range(dataset.num_samples):
    # get prediction
    keypoint_scale, keypoint_xyz21, coord3d_pred_v, coord3d_can_v, keypoint_xyz21_can, R_v, rot_mat = sess.run([data['keypoint_scale'],data['keypoint_xyz21'], coord3d_pred, coord3d_can, data['keypoint_xyz21_can'], R, data['rot_mat']])
    keypoint_xyz21 = np.squeeze(keypoint_xyz21)
    keypoint_scale = np.squeeze(keypoint_scale)
    coord3d_pred_v = np.squeeze(coord3d_pred_v)
    coord3d_can_v = np.squeeze(coord3d_can_v)
    keypoint_xyz21_can = np.squeeze(keypoint_xyz21_can)



    # rescale to meters
    coord3d_pred_v *= keypoint_scale
    # center gt
    keypoint_xyz21 -= keypoint_xyz21[0, :]

    kp_vis = np.ones_like(keypoint_xyz21[:, 0])

    util.feed(keypoint_xyz21, kp_vis, coord3d_pred_v)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

# Output results
mean, median, auc, _, _ = util.get_measures(0.020, 0.050, 20)
print('Evaluation results for %s:' % VARIANT)
print('Average mean EPE: %.3f mm' % (mean*1000))
print('Average median EPE: %.3f mm' % (median*1000))
print('Area under curve: %.3f' % auc)

'''

#gan
for i in range(dataset.num_samples):
    # get prediction

    keypoint_xyz21, coord3d_pred_v, coord3d_can_v, keypoint_xyz21_can, R_v, rot_mat = sess.run([data['keypoint_xyz21'], coord3d_pred, coord3d_can, data['keypoint_xyz21_can'], R, data['rot_mat']])
    keypoint_xyz21 = np.squeeze(keypoint_xyz21)
    coord3d_pred_v = np.squeeze(coord3d_pred_v)
    coord3d_can_v = np.squeeze(coord3d_can_v)
    keypoint_xyz21_can = np.squeeze(keypoint_xyz21_can)


    kp_vis = np.ones_like(keypoint_xyz21[:, 0])

    util.feed(keypoint_xyz21_can, kp_vis, coord3d_can_v)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))



# Output results
mean, median, auc, _, _ = util.get_measures(0.20, 0.50, 20)
print('Evaluation results for %s:' % VARIANT)
print('Average mean EPE: %.3f mm' % (mean*100))
print('Average median EPE: %.3f mm' % (median*100))
print('Area under curve: %.3f' % auc)


'''

curve_list = get_stb_ref_curves()
fig = plt.figure()
ax = fig.add_subplot(111)
for t, v, name in curve_list:
    ax.plot(t, v, label=name)
ax.set_xlabel('threshold in mm')
ax.set_ylabel('PCK')
plt.legend(loc='lower right')
path = './auc_3d_vae.png'
plt.savefig(path,format='png', dpi=300)
#plt.savefig(path,format='png', dpi=300)
plt.close()

   
fig = plt.figure()
ax1= plt.subplot(121,projection='3d')
plot_hand_3d(keypoint_xyz21_can,ax1)
ax1.view_init(azim=-90.0, elev=-90.0)
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 1])
ax1.set_zlim([-3, 3])
ax2 = plt.subplot(122, projection='3d')
plot_hand_3d(coord3d_can_v,ax2)
ax2.view_init(azim=-90.0, elev=-90.0)
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 1])
ax2.set_zlim([-3, 3])
plt.savefig('./gan_org/eval3d_gan_org_%s.png'%str(i),format='png', dpi=300)
plt.close()
'''
