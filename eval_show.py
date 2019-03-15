import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import utils
import visualize
from model import MaskRCNN

# Load weights 不能加载101
# model_path = "./pretrained/resnet101_weights_tf.h5"
model_path = "./pretrained/mask_rcnn_coco.h5"
# model_path = "./pretrained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

from main import TronConfig


class InferenceConfig(TronConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # NUM_CLASSES = 1 + 8
    # for coco is 1+80
    NUM_CLASSES = 1 + 80
config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = MaskRCNN(mode="inference",config=config)

# class_names = ['background','lc','ls','fsp','crb',
#                   'lws','lwd','lys','lyd']
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load weights trained on MS-COCO
model.load_weights(model_path, by_name=True)

val_image_path = "./data/val"
val_image_files = os.listdir(val_image_path)


# fine tune的时候注意exclude掉一些head层因为coco有81个类 而我们有N个类

for ids, val_image_file in enumerate(val_image_files):
    image = skimage.io.imread(val_image_path + "/" + val_image_file)
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(ids, image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])