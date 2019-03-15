import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
from config import Config
from model import MaskRCNN

class TronConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Tron"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # COCO has 80 classes

def main():
    config = TronConfig()
    config.display()


    model = MaskRCNN(mode="training", config=config)

    # Load weights 不能加载101
    # model_path = "./pretrained/resnet101_weights_tf.h5"

    # model.load_weights(model_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    # model_path = "./pretrained/mask_rcnn_coco.h5"

    model_path = "./pretrained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)


    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)


    # Training - Stage 1
    print("Training network heads")
    model.train(learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='head',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='4+', # OOM
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers='all',
                augmentation=augmentation)
if __name__ == "__main__":
    main()