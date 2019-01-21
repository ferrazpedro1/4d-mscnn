#!/bin/bash

clear

echo "======================================================="
echo " MSCNN 3D - Pedro A. P. Ferraz <ferrazpedro1@gmail.com>"
echo "        --- Evaluate MSCNN train/test---               "
echo "======================================================="

# Parameters

kitti_Root='/media/arquivos/deepLearning/KITTI'
mscnn_Root='/home/pedro/deepLearning/3d-mscnn'


rm $kitti_Root/leftCamera/training/image_2


# Car
# --------------------------------
# Testing with image_proc_0_70
# --------------------------------

if [ -d "$mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_0_70" ]; then

	rm $kitti_Root/leftCamera/training/image_2
	ln -s $kitti_Processed/mc-cnn/image_proc_0_70 $kitti_Root/leftCamera/training/image_2

	# Run MSCNN eval

	cd $mscnn_Root/examples/kitti_result/
	matlab -nodisplay -nosplash -r "writeDetForEval('mscnn-8s-768-image_proc_0_70','trainval','0')"
fi

# --------------------------------
# Testing with image_proc_1_70
# --------------------------------

if [ -d "$mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_1_70" ]; then

	rm $kitti_Root/leftCamera/training/image_2	
	ln -s $kitti_Processed/mc-cnn/image_proc_1_70 $kitti_Root/leftCamera/training/image_2

	# Run MSCNN eval

	cd $mscnn_Root/examples/kitti_result/
	matlab -nodisplay -nosplash -r "writeDetForEval('mscnn-8s-768-image_proc_1_70','trainval','0')"
fi


# --------------------------------
# Testing with image_proc_0_80
# --------------------------------

if [ -d "$mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_0_80" ]; then

	rm $kitti_Root/leftCamera/training/image_2	
	ln -s $kitti_Processed/mc-cnn/image_proc_0_80 $kitti_Root/leftCamera/training/image_2

	# Run MSCNN eval

	cd $mscnn_Root/examples/kitti_result/
	matlab -nodisplay -nosplash -r "writeDetForEval('mscnn-8s-768-image_proc_0_80','trainval','0')"
fi

# --------------------------------
# Testing with image_proc_1_80
# --------------------------------

if [ -d "$mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_1_80" ]; then

	rm $kitti_Root/leftCamera/training/image_2	
	ln -s $kitti_Processed/mc-cnn/image_proc_1_80 $kitti_Root/leftCamera/training/image_2

	# Run MSCNN eval

	cd $mscnn_Root/examples/kitti_result/
	matlab -nodisplay -nosplash -r "writeDetForEval('mscnn-8s-768-image_proc_1_80','trainval','0')"
fi




# ------------------------------------
# testing with the original kitti dataset
# ------------------------------------


# Car
# --------------------------------
# Testing with image_proc_0_70
# --------------------------------

	rm $kitti_Root/leftCamera/training/image_2
	ln -s $kitti_Root/leftCamera/training/image_2_ori $kitti_Root/leftCamera/training/image_2

	# Run MSCNN eval

	cd $mscnn_Root/examples/kitti_result/
	matlab -nodisplay -nosplash -r "writeDetForEval('mscnn-8s-768-image_proc_0_70','trainval','ori')"
	matlab -nodisplay -nosplash -r "writeDetForEval('mscnn-8s-768-image_proc_1_70','trainval','ori')"
	matlab -nodisplay -nosplash -r "writeDetForEval('mscnn-8s-768-image_proc_0_80','trainval','ori')"
	matlab -nodisplay -nosplash -r "writeDetForEval('mscnn-8s-768-image_proc_1_80','trainval','ori')"


