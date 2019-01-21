#!/bin/bash

clear

echo "========================================================"
echo " MSCNN 3D - Pedro A. P. Ferraz <ferrazpedro1@gmail.com> "
echo "         --- Batch detection process ---                 "
echo "========================================================"

# Parameters

kitti_Root='/media/arquivos/deepLearning/KITTI'
kitti_Processed='/media/arquivos/KITTIm'
mscnn_Root='/home/pedro/deepLearning/3d-mscnn'


echo " MSCNN - 3d sgmstereo + stereoslic"


cd $mscnn_Root/examples/kitti_car/

# --------------------------------
# Testing with image_proc_0_70
# --------------------------------

	ln -s $kitti_Processed/mccnn/mccnn-70/image_proc_0_70 $kitti_Root/leftCamera/training/image_2
	matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-8s-768-image_proc_0_70')"

# --------------------------------
# Testing with image_proc_1_70
# --------------------------------

	#ln -s $kitti_Processed/mc-cnn/image_proc_1_70 $kitti_Root/leftCamera/training/image_2
	#matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-8s-768-image_proc_1_70')"

# --------------------------------
# Testing with image_proc_0_80
# --------------------------------

	ln -s $kitti_Processed/mccnn/mccnn-70/image_proc_0_80 $kitti_Root/leftCamera/training/image_2        
	matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-8s-768-image_proc_0_80')"

# --------------------------------
# Testing with image_proc_1_80
# --------------------------------

	#ln -s $kitti_Processed/mc-cnn/image_proc_1_80 $kitti_Root/leftCamera/training/image_2
	#matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-8s-768-image_proc_1_80')"

#echo " MSCNN - "
