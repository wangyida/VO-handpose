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
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
#from nets.ColorHandPose3DNetwork_devae import ColorHandPose3DNetwork
from nets.ColorHandPose3DNetwork_org import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d, load_weights_from_snapshot
#import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')image_list = list()
USE_RETRAINED = True
#PATH_TO_POSENET_SNAPSHOTS = '../snapshots/snapshots_posenet_0718_obj+2000/'  # only used when
PATH_TO_POSENET_SNAPSHOTS = '../snapshots/snapshots_posenet_0223_wrist/'  # only used when
#PATH_TO_POSENET_SNAPSHOTS = '../snapshots/snapshots_posenet_0706_wrist/' # USE_RETRAINED is true
#PATH_TO_HANDSEGNET_SNAPSHOTS = '../snapshots/snapshots_handsegnet_devae_0720/150k/'  # only used when USE_RETRAINED is true
PATH_TO_HANDSEGNET_SNAPSHOTS = '../snapshots/snapshots_handsegnet_org_0725/'  # only used when
#PATH_TO_LIFTINGNET_SNAPSHOTS = '../snapshots/snapshots_lifting_proposed_ml_all/'
PATH_TO_LIFTINGNET_SNAPSHOTS = '../snapshots/snapshots_lifting_proposed_wrist/'
#PATH_TO_LIFTINGNET_SNAPSHOTS = '../snapshots/snapshots_un_0221_0d_128/'
vidcap = cv2.VideoCapture('./grasp10.mov')
success,image = vidcap.read()
count = 0
count1 = 0

success = True
image_list = list()
while success:
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
    cv2.imwrite('./data/frvideo10_org/frame%d.jpg'% count, image)     # save frame as JPEG file
    count += 1
    #print(count)
    #if __name__ == '__main__':
        #image_list = list()
    image_list.append('./data/frvideo10_org/frame%d.jpg' %count)

count = 0
count1 = 0

image_list = list()
for count in range (0,3000):
    image_list.append('./data/frvideo10_org/frame%d.jpg'%count)

count1 = 0


# network input
image_tf = tf.placeholder(tf.float32, shape=(1, 320, 320, 3))
hand_side_tf = tf.constant([[0.0, 1.0]])  # right hand 
evaluation = tf.placeholder_with_default(True, shape=())

# build network
net = ColorHandPose3DNetwork()
#hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf,_ = net.inference(image_tf, hand_side_tf, None, evaluation)
hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf,_ = net.inference(image_tf, hand_side_tf, evaluation)
# Start TF
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init)
# initialize network
# net.init(sess)
# initialize network weights
if USE_RETRAINED:
    # retrained version: HandSegNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    print('finishhed')
    # retrained version: PoseNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    print('finishhed')
    last_cpt = tf.train.latest_checkpoint(PATH_TO_LIFTINGNET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    print('finishhed')

else:
    # load weights used in the paper
    net.init(sess)

# Feed image list through network
for img_name in image_list:
    image_raw = scipy.misc.imread(img_name)
    image_raw = scipy.misc.imresize(image_raw, (320, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                             keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                            feed_dict={image_tf: image_v})

    hand_scoremap_v = np.squeeze(hand_scoremap_v)
    image_crop_v = np.squeeze(image_crop_v)
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

      # post processing
    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

        # visualize
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')
    ax1.imshow(image_raw)
    plot_hand(coord_hw, ax1)
    ax2.imshow(image_crop_v)
    plot_hand(coord_hw_crop, ax2)
    ax3.imshow(np.argmax(hand_scoremap_v, 2))
    plot_hand_3d(keypoint_coord3d_v, ax4)
    ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    ax4.set_xlim([-3, 3])
    ax4.set_ylim([-3, 1])
    ax4.set_zlim([-3, 3])
    #plt.show()
    plt.savefig('./results/frvideo10_org/%d.png' %count1)
    plt.close(fig)
    count1 += 1
    #if (count1 == count):
        #break
        #fig.savefig('./results'+img_name)
print("saved all.")

img_root = './results/frvideo10_org/'
fps = 24
print ("here1.")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('saveVideo_sign10_org.avi',fourcc,fps,(640,480))
print ("here.")
for i in range(69):
    frame = cv2.imread(img_root+str(i)+'.png')
    videoWriter.write(frame)
videoWriter.release()

