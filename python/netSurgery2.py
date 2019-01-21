import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()

# Original model VGG - the one we use to copy weights - knowledge transfer
net = caffe.Net('../models/VGG/vgg_train_val.prototxt','../models/VGG/VGG_ILSVRC_16_layers.caffemodel',caffe.TEST)
print("Original VGG Net model:")
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

# MSCNN trained only with dispmap - conv_1_1 has 64 3x3x1 kernels
dispMapNet = caffe.Net('../examples/kitti_car/mscnn-7s-576-2x-dispMap/trainval_1st_xavier.prototxt','../examples/kitti_car/mscnn-7s-576-2x-dispMap/mscnn_kitti_train_1st_iter_40000.caffemodel',caffe.TEST)
print("MSCNN 1st net trained with dispmap only:")
print("blobs {}\nparams {}".format(dispMapNet.blobs.keys(), dispMapNet.params.keys()))


kernels = net.params["conv1_1"][0].data
print("Shape conv1_1: {}".format(np.shape(kernels)))

print("------------------------------------------------------------------------")
print("Part 1 ok. Original model loaded.")
print("------------------------------------------------------------------------")

# saving weights and biases from conv1_1
conv1_1_weights = net.params['conv1_1'][0].data
conv1_1_biases = net.params['conv1_1'][1].data

# creating a new VGG Net with RGBA image input
new_net = caffe.Net('../models/VGG/vgg_3d_train_val.prototxt',caffe.TEST)

# creating new array with transparency
alpha_layer = np.zeros((64,1,3,3))

# Initializing the alpha(transparency) layer as the average of the color RGB channels
for x in xrange(64):
    alpha_layer[x] = np.max([conv1_1_weights[x][0],conv1_1_weights[x][1],conv1_1_weights[x][2]], axis=0)

print "conv1_1"
print conv1_1_weights[0]
print "alpha layer"
print alpha_layer[0]

# Concatenating the original weights and alpha layer
conv1_1_weights_final = np.concatenate((conv1_1_weights,alpha_layer),axis=1)

# Copying the new 3x3x4 arrays to new VGG Net
new_net.params['conv1_1'][0].data[...] = conv1_1_weights_final
new_net.params['conv1_1'][1].data[...] = conv1_1_biases

# Now we need to copy all other layers from the original file to the new_net

# data
#new_net.params['input'][0].data[...] = net.params['input'][0].data
#new_net.params['input'][1].data[...] = net.params['input'][1].data

# conv1_2
new_net.params['conv1_2'][0].data[...] = net.params["conv1_2"][0].data
new_net.params['conv1_2'][1].data[...] = net.params["conv1_2"][1].data

# conv2_1
new_net.params['conv2_1'][0].data[...] = net.params["conv2_1"][0].data
new_net.params['conv2_1'][1].data[...] = net.params["conv2_1"][1].data

# conv2_2
new_net.params['conv2_2'][0].data[...] = net.params["conv2_2"][0].data
new_net.params['conv2_2'][1].data[...] = net.params["conv2_2"][1].data

# conv3_1
new_net.params['conv3_1'][0].data[...] = net.params["conv3_1"][0].data
new_net.params['conv3_1'][1].data[...] = net.params["conv3_1"][1].data

# conv3_2
new_net.params['conv3_2'][0].data[...] = net.params["conv3_2"][0].data
new_net.params['conv3_2'][1].data[...] = net.params["conv3_2"][1].data

# conv3_3
new_net.params['conv3_3'][0].data[...] = net.params["conv3_3"][0].data
new_net.params['conv3_3'][1].data[...] = net.params["conv3_3"][1].data

# conv4_1
new_net.params['conv4_1'][0].data[...] = net.params["conv4_1"][0].data
new_net.params['conv4_1'][1].data[...] = net.params["conv4_1"][1].data

# conv4_2
new_net.params['conv4_2'][0].data[...] = net.params["conv4_2"][0].data
new_net.params['conv4_2'][1].data[...] = net.params["conv4_2"][1].data

# conv4_3
new_net.params['conv4_3'][0].data[...] = net.params["conv4_3"][0].data
new_net.params['conv4_3'][1].data[...] = net.params["conv4_3"][1].data

# conv5_1
new_net.params['conv5_1'][0].data[...] = net.params["conv5_1"][0].data
new_net.params['conv5_1'][1].data[...] = net.params["conv5_1"][1].data

# conv5_2
new_net.params['conv5_2'][0].data[...] = net.params["conv5_2"][0].data
new_net.params['conv5_2'][1].data[...] = net.params["conv5_2"][1].data

# conv5_3
new_net.params['conv5_3'][0].data[...] = net.params["conv5_3"][0].data
new_net.params['conv5_3'][1].data[...] = net.params["conv5_3"][1].data

# Not sure if I need to copy the FC layers (probably not) but just in case ;)
# fc6
new_net.params['fc6'][0].data[...] = net.params["fc6"][0].data
new_net.params['fc6'][1].data[...] = net.params["fc6"][1].data

# fc7
new_net.params['fc7'][0].data[...] = net.params["fc7"][0].data
new_net.params['fc7'][1].data[...] = net.params["fc7"][1].data

# fc8
new_net.params['fc8'][0].data[...] = net.params["fc8"][0].data
new_net.params['fc8'][1].data[...] = net.params["fc8"][1].data

# saving new weights
new_net.save('../models/VGG/VGG_RGBA_max_16_layers.caffemodel')

#net = caffe.Net('../models/VGG/vgg_3d_train_val.prototxt','../models/VGG/VGG_RGBA_2_16_layers.caffemodel',caffe.TEST)
