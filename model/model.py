from matterport.config import Config
from matterport import utils
from preprocessing import conversion
import matterport.model as modellib
from preprocessing import normalization

import os
import numpy as np
from utils import resources
from utils.logger import ProgressLogger


class Model:
    def __init__(self, mode):
        self.ROOT_DIR = resources.root
        self.MODEL_DIR = r"C:\Users\MoritzWollenhaupt\Desktop\Masterarbeit\resources\logs2"
        self.COCO_MODEL_PATH = resources.backbone
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
        assert mode in ["training", "inference"]
        self.mode = mode
        if self.mode == "training":
            self.config = RoofTypeConfig()
            self.model = modellib.MaskRCNN(mode=self.mode, config=self.config, model_dir=self.MODEL_DIR)
        if self.mode == "inference":
            self.config = InferenceConfig()
            self.model = modellib.MaskRCNN(mode=self.mode, config=self.config, model_dir=self.MODEL_DIR)

    def train(self, dataset_train, dataset_val, epochs, init_with, layers,
              exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]):
        if self.mode != "training":
            raise Exception("Your model is not in training mode!")
        if init_with == "imagenet":
            self.model.load_weights(self.model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            self.model.load_weights(self.COCO_MODEL_PATH, by_name=True, exclude=exclude)
        elif init_with == "last":
            self.model.load_weights(self.model.find_last(), by_name=True)
        else:
            self.model.load_weights(init_with, by_name=True)

        if layers == "all":
            self.model.train(dataset_train, dataset_val, learning_rate=self.config.LEARNING_RATE / 10, epochs=epochs,
                             layers=layers)
        else:
            self.model.train(dataset_train, dataset_val, learning_rate=self.config.LEARNING_RATE, epochs=epochs,
                             layers=layers)

    def detect(self, model_path, images, type="chip"):
        assert type in ["ndom", "chip"]
        if model_path == "last":
            model_path = self.model.find_last()
        self.model.load_weights(model_path, by_name=True)
        results = []
        splits = []
        if type == "ndom":
            splits.extend(conversion.split_ndom(images[0], True))
        elif type == "chip":
            splits.append(images[0])
        for image in images[1:]:
            if type == "ndom":
                splits.extend(conversion.split_ndom(image))
            elif type == "chip":
                splits.append(image)
        for img in splits:
            image_enrich = normalization.normalize(img)
            result = self.model.detect([image_enrich], verbose=1)[0]
            results.append(result)
        return results, splits

    def evaluate(self, dataset_test, set_size):
        image_ids = np.random.choice(dataset_test.image_ids, set_size)
        APs = []
        logger = ProgressLogger(set_size, self.evaluate)
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset_test, self.config,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, self.config), 0)
            # Run object detection
            results = self.model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
            logger.log_progress()
        return np.mean(APs)


# train-config
class RoofTypeConfig(Config):
    NAME = "roof_types"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT

    LEARNING_RATE = 0.001
    DETECTION_MAX_INSTANCES = 512
    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + x roof types
    # Use small images for faster training.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 3

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 25
    MEAN_PIXEL = np.array([30.4, 127.07, 127.58])
    

    BACKBONE_STRIDES = [4, 8, 16, 32, 64, 128]
    RPN_ANCHOR_SCALES = (10, 20, 40, 80, 160, 320)
    RPN_ANCHOR_RATIOS = [1/3, 0.5, 1, 2, 3]
    RPN_NMS_THRESHOLD = 0.8
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512
    LEARNING_RATE = 0.001
    TRAIN_ROIS_PER_IMAGE = 512
    

# inference-config
class InferenceConfig(RoofTypeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.55
    DETECTION_NMS_THRESHOLD = .3
