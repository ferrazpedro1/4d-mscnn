import glob
import os

outFile = open("result.txt","wa")
kittiGTPath="/home/pedro/deepLearning/SubCNN/fast-rcnn/output/kitti/kitti_val/caffenet_fast_rcnn_msr_kitti_iter_40000/"

os.chdir(kittiGTPath)

for f in sorted(glob.glob("*.txt")):
	# opening input file
	inFile = open(f,"r") 
	for line in inFile:
		outFile.write(inFile.name[:-4]+' '+line)
	inFile.close()
outFile.close()
