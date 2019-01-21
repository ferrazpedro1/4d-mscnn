#!/bin/bash

clear

echo "========================================================"
echo " MSCNN 3D - Pedro A. P. Ferraz <ferrazpedro1@gmail.com> "
echo "         --- Batch training process ---                 "
echo "========================================================"

# Parameters

kitti_Root='/media/arquivos/deepLearning/KITTI'
kitti_Processed='/media/arquivos/BKP_KITTI_CONFERIR/KITTI-Processados'
mscnn_Root='/home/pedro/deepLearning/3d-mscnn'


rm $kitti_Root/leftCamera/training/image_2

echo " MSCNN - 3d sgmstereo + stereoslic"

# --------------------------------
# Testing with image_proc_0_70
# --------------------------------

if [ ! -d "$mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_0_70" ]; then
	
	ln -s $kitti_Processed/mc-cnn/image_proc_0_70 $kitti_Root/leftCamera/training/image_2
	cp -R $mscnn_Root/examples/kitti_car/mscnn-8s-768-trainval-ori/ $mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_0_70/

	# Train MSCNN network
	cd $mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_0_70
	sh train_mscnn.sh

	# Run MSCNN detection - After training the network
	cd ..
	matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-8s-768-image_proc_0_70')"

	# Create a tar.gz file with backup
	tar -zcvf mscnn-8s-768-image_proc_0_70.tar.gz mscnn-8s-768-image_proc_0_70/

fi

# --------------------------------
# Testing with image_proc_1_70
# --------------------------------

if [ ! -d "$mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_1_70" ]; then
	
	ln -s $kitti_Processed/mc-cnn/image_proc_1_70 $kitti_Root/leftCamera/training/image_2
	cp -R $mscnn_Root/examples/kitti_car/mscnn-8s-768-trainval-ori/ $mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_1_70/

	# Train MSCNN network
	cd $mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_1_70
	sh train_mscnn_1.sh

	# Run MSCNN detection - After training the network
	cd ..
	matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-8s-768-image_proc_1_70')"

	# Create a tar.gz file with backup
	tar -zcvf mscnn-8s-768-image_proc_1_70.tar.gz mscnn-8s-768-image_proc_1_70/

fi

# --------------------------------
# Testing with image_proc_0_80
# --------------------------------

if [ ! -d "$mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_0_80" ]; then

        ln -s $kitti_Processed/mc-cnn/image_proc_0_80 $kitti_Root/leftCamera/training/image_2
        cp -R $mscnn_Root/examples/kitti_car/mscnn-8s-768-trainval-ori/ $mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_0_80/

        # Train MSCNN network
        cd $mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_0_80
        sh train_mscnn.sh

        # Run MSCNN detection - After training the network
        cd ..
        matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-8s-768-image_proc_0_80')"

        # Create a tar.gz file with backup
        tar -zcvf mscnn-8s-768-image_proc_0_80.tar.gz mscnn-8s-768-image_proc_0_80/

fi

# --------------------------------
# Testing with image_proc_1_80
# --------------------------------

if [ ! -d "$mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_1_80" ]; then
	
	ln -s $kitti_Processed/mc-cnn/image_proc_1_80 $kitti_Root/leftCamera/training/image_2
	cp -R $mscnn_Root/examples/kitti_car/mscnn-8s-768-trainval-ori/ $mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_1_80/

	# Train MSCNN network
	cd $mscnn_Root/examples/kitti_car/mscnn-8s-768-image_proc_1_80
	sh train_mscnn_1.sh

	# Run MSCNN detection - After training the network
	cd ..
	matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-8s-768-image_proc_1_80')"

	# Create a tar.gz file with backup
	tar -zcvf mscnn-8s-768-image_proc_1_80.tar.gz mscnn-8s-768-image_proc_1_80/

fi


