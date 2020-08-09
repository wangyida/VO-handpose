# VO-handpose
--------------------------
@@HAND OBJECT POSE DATASET - version @@x.x
--------------------------

_______
LICENCE
This dataset is provided for research purposes only and without any warranty. Any commercial use is prohibited. If you use the dataset or parts of it in your research, you must cite the respective paper.

@article{gao2019variational,
  title={Variational Object-Aware 3-D Hand Pose From a Single RGB Image},
  author={Gao, Yafei and Wang, Yida and Falco, Pietro and Navab, Nassir and Tombari, Federico},
  journal={IEEE Robotics and Automation Letters},
  volume={4},
  number={4},
  pages={4239--4246},
  year={2019},
  publisher={IEEE}
}
_______
CONTENT

This dataset provides 11020 samples. Each sample provides:
	- RGB image (320x320 pixels); 
	- Segmentation mask (320x320 pixels) for hand
	- Segmentation mask (320x320 pixels) for object 
	- 21 Keypoints for hand with their uv coordinates in the image frame and their xyz coordinates in the camera coordinate system
	- Intrinsic Camera Matrix

It was created with @@freely available character from MakeHuman and rendered with www.blender.org
_______
HOW TO USE

The dataset ships with minimal examples, that browse the dataset and show samples.
There is one example for Phython and one for MATLAB users; Their functionality is identical.
Both files are located in the root folder.

_______
STRUCTURE

./ 			: Root folder
./color			: Color images
./mask_hand		: Segmentation masks for hand
./mask_object		: Segmentation masks for object
./annotation.mat	: Data structure for MATLAB use containing keypoint annotations and camera matrices
_______
CONTACT

For questions about the dataset please contact Yafei Gao (@@yafei.gao@tum.de)
