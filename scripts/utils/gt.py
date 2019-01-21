import glob
import os

outFile = open("result.txt","wa")
kittiGTPath="/home/pedro/deepLearning/KITTI/leftCamera/training/label_2/"

os.chdir(kittiGTPath)

for f in sorted(glob.glob("*.txt")):
	# opening input file
	inFile = open(f,"r") #opens file with name of "test.txt"
	for line in inFile:
		outFile.write(inFile.name[:-4]+' '+line)
	inFile.close()
outFile.close()
