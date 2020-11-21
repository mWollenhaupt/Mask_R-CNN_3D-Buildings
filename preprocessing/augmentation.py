from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters as iaa
import skimage.external.tifffile
import skimage.io
import numpy as np
import shutil
import os

from utils.logger import ProgressLogger


class ImageSet:
    def __init__(self, path, augmented=False):
        self.root = path
        self.augmented = augmented
        self.img = os.path.join(path, "{}.tif".format(os.path.basename(path)))
        self.masks, self.masks_count = self.read_masks()

    def __str__(self):
        return self.root

    def __repr__(self):
        return self.__str__()

    def read_masks(self):
        mask_dirs = [os.path.join(self.root, mask_dir) for mask_dir in os.listdir(self.root) if
                     os.path.isdir(os.path.join(self.root, mask_dir))]
        dct_masks = {}
        dct_masks_count = {}
        for mask_dir in mask_dirs:
            basename = os.path.basename(mask_dir)
            dct_masks[basename] = []
            masks = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if
                     os.path.isfile(os.path.join(mask_dir, file))]
            for mask in masks:
                dct_masks[basename].append(mask)
            dct_masks_count[basename] = len(masks)
        return dct_masks, dct_masks_count

    def clean_invalid_masks(self, min_px=20):
        for key, value in self.masks.items():
            for mask in value:
                val = skimage.io.imread(mask)
                if np.count_nonzero(val == 255) <= 20:
                    os.remove(mask)
                    self.masks[key].remove(mask)
                    self.masks_count[key] -= 1
        return self.verify()

    # check for chips without features in it
    def verify(self):
        keys_to_delete = []
        for key, value in self.masks_count.items():
            if value == 0:
                keys_to_delete.append(key)
                shutil.rmtree(os.path.join(self.root, key))
        for key in keys_to_delete:
            del self.masks[key]
            del self.masks_count[key]
        if len(self.masks) == 0:
            print("lösche vollständig: ", self.root)
            shutil.rmtree(self.root)
            return self
        return None

    def get_count(self, cls):
        if cls in self.masks_count:
            return self.masks_count[cls]
        return 0

    # Augment this object using a passed sequential object
    def augment(self, path, seq):
        def stack_masks():
            masks = []
            types = []
            for key, value in self.masks.items():
                for val in value:
                    masks.append(skimage.io.imread(val))
                    types.append(key)
            stack = np.dstack(masks)
            return stack, types

        masks, types = stack_masks()
        image = skimage.io.imread(self.img)
        segmap = SegmentationMapsOnImage(masks, shape=image.shape)
        image_aug, segmaps_aug = seq(image=image, segmentation_maps=segmap)
        masks_aug = segmaps_aug.get_arr()
        naming_dct = {}
        min_one_mask = False
        for j in range(masks_aug.shape[2]):
            mask = masks_aug[:, :, j]
            if (len(np.unique(mask))) <= 1:  # Background only
                continue
            mtype = types[j]
            if mtype in naming_dct:
                naming_dct[mtype] += 1
            else:
                naming_dct[mtype] = 1
            mpath = os.path.join(path, mtype, "{}.tif".format(str(naming_dct[mtype]).zfill(9)))
            os.makedirs(os.path.dirname(mpath), exist_ok=True)
            with skimage.external.tifffile.TiffWriter(mpath) as tif:
                tif.save(mask, compress=9)
                min_one_mask = True
        if min_one_mask:
            basename = os.path.basename(path)
            ipath = os.path.join(path, "{}.tif".format(basename))
            with skimage.external.tifffile.TiffWriter(ipath) as tif:
                tif.save(image_aug, compress=9)
            return ImageSet(os.path.dirname(ipath), augmented=True)
        return None

# mapping of a complete data set
class ImageSetCollection:
    def __init__(self, path):
        self.path = path
        self.augmented = self.read_augmented()
        self.image_sets = self.read_path()
        self.sometimes = lambda aug: iaa.Sometimes(0.8, aug)
        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                iaa.LinearContrast((0.75, 1.5)),
                # crop images by -10% to 10% of their height/width
                self.sometimes(iaa.CropAndPad(
                    percent=(-0.1, 0.1),
                    pad_cval=0
                )),
                self.sometimes(iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -20 to +20 percent (per axis)
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-175, 175),  # rotate by -175 to +175 degrees
                    shear=(-10, 10),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbor or bilinear interpolation (fast)
                    cval=0,  # if mode is constant, use a cval = 0
                ))
            ], random_order=True)

    def read_augmented(self):
        aug = {}
        try:
            with open(os.path.join(self.path, "aug.txt"), "r") as file:
                lines = [line.split() for line in file.readlines()]
            for line in lines:
                if line[1] == "True":
                    aug[line[0]] = True
                else:
                    aug[line[0]] = False
        except Exception as e:
            pass
        return aug

    def write_augmented(self):
        with open(os.path.join(self.path, "aug.txt"), "w+") as file:
            for image_set in self.image_sets:
                file.write("{}\t{}\n".format(os.path.basename(image_set.root), image_set.augmented))

    def write_stats(self):
        num_images = len(self.image_sets)
        num_features = 0
        features = {}
        images = {}
        for image_set in self.image_sets:
            for key, value in image_set.masks_count.items():
                if key in features:
                    features[key] += value
                    num_features += value
                else:
                    features[key] = value
                    num_features += value
                if key in images:
                    images[key] += 1
                else:
                    images[key] = 1
        num_classes = len(features)
        with open(os.path.join(self.path, "stats.txt"), "w+") as file:
            file.write("images = {}\n".format(num_images))
            file.write("features = {}\n".format(num_features))
            file.write("classes = {}\n".format(num_classes))
            file.write("cls name\t\t\t\timages\t\t\t\t\t\tfeatures\n")
            features = [[key, value] for key, value in features.items()]
            features = sorted(features, key=lambda x: x[0])
            for feature in features:
                file.write("{}\t\t\t\t\t{}\t\t\t\t\t{}\n".format(feature[0], str(images[feature[0]]).zfill(8),
                                                                 str(feature[1]).zfill(8)))

    def read_path(self):
        image_set_paths = [os.path.join(self.path, folder) for folder in os.listdir(self.path) if
                           os.path.isdir(os.path.join(self.path, folder))]
        image_sets = []
        logger = ProgressLogger(len(image_set_paths), self.read_path)
        for path in image_set_paths:
            basename = os.path.basename(path)
            if basename in self.augmented:
                aug = self.augmented[basename]
            else:
                aug = False
            image_sets.append(ImageSet(path, augmented=aug))
            logger.log_progress()
        return image_sets

    def clean_invalid_masks(self, min_px=20):
        logger = ProgressLogger(len(self.image_sets), self.clean_invalid_masks)
        for image_set in self.image_sets:
            res = image_set.clean_invalid_masks(min_px=min_px)
            if res is not None:
                self.image_sets.remove(res)
            logger.log_progress()
        self.write_augmented()
        self.write_stats()

    def verify(self):
        logger = ProgressLogger(len(self.image_sets), self.verify)
        for image_set in self.image_sets:
            res = image_set.verify()
            if res is not None:
                self.image_sets.remove(res)
            logger.log_progress()
        self.write_augmented()
        self.write_stats()

    def search_by_masks(self, classes, gt=None):
        if gt is None:
            gt = [0]*len(classes)
        matches = []
        for i in range(len(self.image_sets)):
            image_set = self.image_sets[i]
            tmp = True
            for j in range(len(classes)):
                cls = classes[j]
                if image_set.get_count(cls) <= gt[j]:
                    tmp = False
            if tmp:
                matches.append(i)
        return matches

    def delete_image_sets_by_idx(self, idx):
        to_delete = [self.image_sets[i] for i in idx]
        logger = ProgressLogger(len(to_delete), self.delete_image_sets_by_idx)
        for image_set in to_delete:
            shutil.rmtree(image_set.root)
            self.image_sets.remove(image_set)
            logger.log_progress()
        self.write_augmented()
        self.write_stats()

    def delete_image_sets_with_mask(self, cls, gt=0):
        to_delete = []
        for image_set in self.image_sets:
            if image_set.get_count(cls) > gt:
                to_delete.append(image_set)
        for image_set in to_delete:
            shutil.rmtree(image_set.root)
            self.image_sets.remove(image_set)
        self.write_augmented()
        self.write_stats()

    def augment(self, ids, seq=None, iterations=1):
        if seq is None:
            seq = self.seq
        idx = int(sorted(list(self.augmented))[-1])+1
        logger = ProgressLogger(len(ids)*iterations, self.augment)
        for j in ids:
            image_set = self.image_sets[j]
            if image_set.augmented:
                continue
            for i in range(iterations):
                path = os.path.join(self.path, str(idx).zfill(9))
                aug = image_set.augment(path, seq)
                if aug is not None:
                    self.image_sets.append(aug)
                    idx += 1
                logger.log_progress()
        self.write_augmented()
        self.augmented = self.read_augmented()
        self.write_stats()
