#!/bin/bash

clear

echo "========================================================"
echo " MSCNN 3D - Pedro A. P. Ferraz <ferrazpedro1@gmail.com> "
echo "         --- Batch training process ---                 "
echo "         Using non-linear disparity map                 "   
echo " ------------------------------------------------------ "
echo " Parameters: 					      "
echo "  MCCNN: 80 pixels disparity			      "
echo "  Power Transform [pixelFinal=c*pixelOriginal^(y)]      "
echo "     c=255 and y=0.2/0.4/0.6			      "
echo "  DispMap combined using addWeighted                    "
echo "     (alpha=0.7 and gamma=-10)     	              "
echo "========================================================"

# Parameters

kitti_Root='/media/arquivos/deepLearning/KITTI'
kitti_Processed='/media/arquivos/KITTIm//mccnn/mccnn-80'
mscnn_Root='/home/pedro/deepLearning/3d-mscnn'


rm $kitti_Root/leftCamera/training/image_2

echo " MSCNN - MCCNN -> power transform (255/0.4)-> addWeighted"


#if [ ! -d "$mscnn_Root/examples/kitti_car/mscnn-7s-576-2x-ori-80-nonLinear_255-0.4/" ]; then
#	
#	ln -s $kitti_Processed/image_proc_0_80_nonLinear_255_0.4/ $kitti_Root/leftCamera/training/image_2
#	cp -R $mscnn_Root/examples/kitti_car/mscnn-7s-576-2x-ori/ $mscnn_Root/examples/kitti_car/mscnn-7s-576-2x-ori-80-nonLinear_255-0.4/
#
#	# Train MSCNN network
#	cd $mscnn_Root/examples/kitti_car/mscnn-7s-576-2x-ori-80-nonLinear_255-0.4/
#	sh train_mscnn.sh
#
#	# Run MSCNN detection - After training the network
#	cd ..
#	matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-7s-576-2x-ori-80-nonLinear_255-0.4/')"
#
#	# Create a tar.gz file with backup
#	tar -zcvf mscnn-7s-576-2x-ori-80-nonLinear_255-0.4/ mscnn-7s-576-2x-ori-80-nonLinear_255-0.4/
#
#fi

# -------------------------------------------------------------------------------------------------------------------------------------------

if [ ! -d "$mscnn_Root/examples/kitti_car/mscnn-7s-576-2x-1-80-nonLinear_255-0.4/" ]; then

        ln -s $kitti_Processed/image_proc_1_80_nonLinear_255_0.4/ $kitti_Root/leftCamera/training/image_2
        cp -R $mscnn_Root/examples/kitti_car/mscnn-7s-576-2x-ori/ $mscnn_Root/examples/kitti_car/mscnn-7s-576-2x-1-80-nonLinear_255-0.4/

        # Train MSCNN network
        cd $mscnn_Root/examples/kitti_car/mscnn-7s-576-2x-1-80-nonLinear_255-0.4/
        sh train_mscnn.sh

        # Run MSCNN detection - After training the network
        cd ..
        matlab -nojvm -nodisplay -nosplash -r "run_mscnn_detection_3d('mscnn-7s-576-2x-1-80-nonLinear_255-0.4/')"

        # Create a tar.gz file with backup
        tar -zcvf mscnn-7s-576-2x-1-80-nonLinear_255-0.4/ mscnn-7s-576-2x-ori-1-nonLinear_255-0.4/

fi


