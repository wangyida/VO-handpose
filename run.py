
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
# np.set_printoptions(threshold=np.nan)
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from nets.ColorHandPose3DNetwork_org import ColorHandPose3DNetwork
from nets.ColorHandPose3DNetwork_devae import ColorHandPose3DNetwork
#from nets.ColorHandPose3DNetwork_pose2d import ColorHandPose3DNetwork 
from pca import pca
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d, load_weights_from_snapshot
import os
USE_RETRAINED = True
PATH_TO_POSENET_SNAPSHOTS = '/home/gaoyafei/original/thesis/snapshots_posenet_0223_wrist/'
#PATH_TO_POSENET_SNAPSHOTS = '../snapshots/snapshots_posenet_0724/'  # only used when USE_RETRAINED is true
#PATH_TO_POSENET_SNAPSHOTS = '../snapshots/snapshots_posenet_hg5/'
#PATH_TO_HANDSEGNET_SNAPSHOTS = '../snapshots/snapshots_handsegnet_0716_all/'  # only used when USE_RETRAINED is true
PATH_TO_HANDSEGNET_SNAPSHOTS = '/home/gaoyafei/original/thesis/snapshots_handsegnet_devae_0720/150k/'  # only used when USE_RETRAINED is true
#PATH_TO_LIFTING_SNAPSHOTS  = '../snapshots/snapshots_lifting_proposed_wrist/'
#PATH_TO_LIFTING_SNAPSHOTS  = '../final_snapshots/snapshots_lifting_proposed_ml_all/'
PATH_TO_LIFTING_SNAPSHOTS  = '/home/gaoyafei/original/thesis/snapshots_lifting_0731_wnl/'
#infile = open('/home/gaoyafei/Desktop/img_coord3d_can.txt','w')
if __name__ == '__main__':
    # images to be shown
    image_list = []


    #g = os.walk('/media/wangyida/D0-P1/finished_0620/color')
    #g1 = os.walk('/media/wangyida/D0-P1/finished_0611/test_realhand')
    #g = os.walk('/home/gaoyafei/Downloads/new_real/')
    g = os.walk('/media/wangyida/HDD/finished_0614/test/color')    
    '''
    for path,dir_list,file_list in g1:  
        for file_name in file_list:  
            img_path = str(path)+'/'+str(file_name)
            #print(img_path) 
            image_list.append(img_path)
    '''
    for path,dir_list,file_list in g:
        for file_name in file_list:
            img_path = str(path)+'/'+str(file_name)
            image_list.append(img_path)
    image_tf = tf.placeholder(tf.float32, shape=(1, 320, 320, 3))
    hand_side_tf = tf.constant([[0.0, 1.0]])  # right hand
    #hand_side_tf = tf.constant([[1.0, 0.0]]) # left hand
    evaluation = tf.placeholder_with_default(True, shape=())
    # build network
    net = ColorHandPose3DNetwork()
    #hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,keypoints_scoremap_tf, keypoint_coord3d_tf,_ = net.inference(image_tf, hand_side_tf,evaluation)
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,keypoints_scoremap_tf, keypoint_coord3d_tf, coord_can_tf, latent_tf, nebula3d_tf = net.inference(image_tf, hand_side_tf,None,evaluation)   #latent->@@latent     
    # Start TF
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    cmp_list = [44,73,86]
    # initialize network
    if (USE_RETRAINED):
        
        # retrained version: HandSegNet
        last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
                
        # retrained version: PoseNet
        last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam','global_step', 'beta'])
        '''
        last_cpt = tf.train.latest_checkpoint(PATH_TO_2D_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
        '''
        # retrained version: LiftingNet
        last_cpt = tf.train.latest_checkpoint(PATH_TO_LIFTING_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
        
    else:
        # load weights used in the paper
        net.init(sess)
    latent = []
    nebula3d = []
    ig_name = 0
    obj_label = []
    for img_name in image_list:

        name_string = img_name.split('/')
        name = name_string[-1]
        ig_name += 1
        image_raw = scipy.misc.imread(img_name)
        label = int(name.split('_')[0][1])
        obj_label.append(label)
        print(img_name)


        image_raw = scipy.misc.imresize(image_raw, (320, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)
        image_v = image_v[:,:,:,:3]
        hand_scoremap_v, image_crop_v, scale_v, center_v, keypoints_scoremap_v, keypoint_coord3d_v, coord_can_v, latent_v, nebula3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,keypoints_scoremap_tf, keypoint_coord3d_tf, coord_can_tf, latent_tf, nebula3d_tf], feed_dict={image_tf: image_v})  #latent->@@latent
        latent = np.append(latent, latent_v)
        #nebula3d = [nebula3d, nebula3d_v]
        #hand_scoremap_v, image_crop_v, scale_v, center_v, keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,keypoints_scoremap_tf, keypoint_coord3d_tf], feed_dict={image_tf: image_v})

        hand_scoremap_v = np.squeeze(hand_scoremap_v) 
        image_crop_v = np.squeeze(image_crop_v)                                                                                                                                             
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        keypoint_coord3d_can_v = np.squeeze(coord_can_v)
        keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

        image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

        # visualize
        if 1:#ig_name == 68:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            ax1.imshow(image_raw)
            plot_hand(coord_hw, ax1)
            plt.savefig('./result_0224_ours_test/%s_1.png'%ig_name,bbox_inches = 'tight')
            plt.close()

            fig = plt.figure()
            ax2 = fig.add_subplot(111)
            #plt.xticks([])
            #plt.yticks([])
            #plt.axis('off')
            #ax2.imshow(image_crop_v)
            #plot_hand(coord_hw_crop, ax2)
            ax2 = plt.subplot(222,projection='3d')
            plot_hand_3d(keypoint_coord3d_can_v,ax2)
            ax2.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            ax2.set_xlim([-3, 3])
            ax2.set_ylim([-3, 1])
            ax2.set_zlim([-3, 3])
            plt.savefig('./result_0224_ours_test/%s_2.png'%ig_name,bbox_inches = 'tight')
            plt.close()

            fig = plt.figure()
            ax3 = fig.add_subplot(111)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            ax3.imshow(np.argmax(hand_scoremap_v,2))
            plt.savefig('./result_0224_ours_test/%s_3.png'%ig_name,bbox_inches = 'tight')
            plt.close()

            fig = plt.figure()
            ax4 = fig.add_subplot(111, projection='3d')
            plot_hand_3d(keypoint_coord3d_v, ax4)
            ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            ax4.set_xlim([-3, 3])
            ax4.set_ylim([-3, 1])
            ax4.set_zlim([-3, 3])
            plt.savefig('./result_0224_ours_test/%s_4.png'%ig_name,bbox_inches = 'tight')
            plt.close()

        if 0:
            fig = plt.figure()
            ax1 = plt.subplot(221)
            plt.imshow(image_raw)
            plot_hand(coord_hw, ax1)
            ax2 = plt.subplot(222)
            plt.imshow(image_crop_v)
            plot_hand(coord_hw_crop, ax2)
            ax3 = plt.subplot(223)
            plt.imshow(np.argmax(hand_scoremap_v,2))
            ax4 = plt.subplot(224,projection='3d')
            plot_hand_3d(keypoint_coord3d_v,ax4)
            ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            ax4.set_xlim([-3, 3])
            ax4.set_ylim([-3, 1])
            ax4.set_zlim([-3, 3])
            path = './result_0224_ours/'+name
            plt.savefig(path,format='png', dpi=300)
            plt.close()
        if 0:
            fig = plt.figure()
            ax1 = plt.subplot(221)
            plt.imshow(image_raw)
            plot_hand(coord_hw, ax1)
            plt.axis('off')
            ax2 = plt.subplot(222,projection='3d')
            plot_hand_3d(keypoint_coord3d_can_v,ax2)
            ax2.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            ax2.set_xlim([-3, 3])
            ax2.set_ylim([-3, 1])
            ax2.set_zlim([-3, 3])
            ax3 = plt.subplot(223)
            plt.imshow(np.argmax(hand_scoremap_v,2))
            plt.axis('off')
            ax4 = plt.subplot(224,projection='3d')
            plot_hand_3d(keypoint_coord3d_v,ax4)
            ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            ax4.set_xlim([-3, 3])
            ax4.set_ylim([-3, 1])
            ax4.set_zlim([-3, 3])
            path = './result_0224_ours_train/'+name
            plt.savefig(path,format='png', dpi=300)
            plt.close()




    latent = np.reshape(latent, (-1, 256))    
    _, V = pca(latent, dim_remain=2)

    latent_viz = np.matmul(latent, V)
    nebula_viz = np.matmul(nebula3d_v, V)
    dist = np.zeros((latent_viz.shape[0],nebula_viz.shape[0]))

    for i in range (0, latent_viz.shape[0]):
        for j in range(0,nebula_viz.shape[0]):
            dist[i,j] = np.linalg.norm(latent_viz[i,:]-nebula_viz[j,:])
     
    index = np.argmin(dist, 1)

    info = np.vstack((obj_label, index, latent_viz[:,0],latent_viz[:,1], dist[:,1],dist[:,2],dist[:,3],dist[:,4] )).T

    print (info)

        
