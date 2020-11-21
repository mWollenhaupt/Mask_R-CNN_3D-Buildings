from utils.multiprocessing import ThreadPool
from utils.multiprocessing import split
from utils import resources
from utils.logger import ProgressLogger
from preprocessing import normalization
from preprocessing.augmentation import ImageSetCollection
import skimage.io
import skimage.external.tifffile
import numpy as np
import shutil
import arcpy
import os


# Use a moving window to generate chips of a given size
def create_training_chips(images, chips_dir, shapefile, value_field, tile_size, stride, rotation=0):
    logger = ProgressLogger(len(images), create_training_chips)
    for image in images:
        arcpy.sa.ExportTrainingDataForDeepLearning(in_raster=image, out_folder=chips_dir,
                                                   in_class_data=shapefile, image_chip_format="TIFF",
                                                   tile_size_x=tile_size, tile_size_y=tile_size,
                                                   output_nofeature_tiles="ONLY_TILES_WITH_FEATURES",
                                                   stride_x=stride, stride_y=stride, metadata_format="RCNN_Masks",
                                                   rotation_angle=rotation, class_value_field=value_field)
        logger.log_progress()

# The multiprocessing approach has no advantage here. Do not use!
def create_training_chips_mt(images_dir, chips_dir, shapefile, value_field, tile_size, stride, rotation=0,
                             num_threads=1):
    def merge_results(chips_dir):
        os.makedirs(os.path.join(chips_dir, "images"), exist_ok=True)
        counter = 0
        mapping = []
        for sub_chip in [os.path.join(resources.temp_chips, sub) for sub in os.listdir(resources.temp_chips) if
                         os.path.isdir(os.path.join(resources.temp_chips, sub))]:
            with open(os.path.join(sub_chip, "map.txt"), "r") as file:
                lines = file.readlines()
            for line in lines:
                row = []
                split = line.split()
                image = os.path.join(sub_chip, split[0])
                labels = [os.path.join(sub_chip, label) for label in split[1:]]
                shutil.copy(image, os.path.join(chips_dir, "images", "{}.tif".format(str(counter).zfill(9))))
                row.append("images\\{}.tif  ".format(str(counter).zfill(9)))
                for label in labels:
                    class_id = os.path.split(os.path.split(label)[0])[1]
                    path = os.path.join(chips_dir, "labels", class_id)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    shutil.copy(label, os.path.join(path, "{}.tif".format(str(counter).zfill(9))))
                    row.append("labels\\{}\\{}.tif ".format(class_id, str(counter).zfill(9)))
                counter += 1
                mapping.append(row)
        with open(os.path.join(chips_dir, "map.txt"), "w+") as file:
            for row in mapping:
                for element in row:
                    file.write(element)
                file.write("\n")

    os.makedirs(resources.temp_chips, exist_ok=True)
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if image.endswith(".tif")]
    images = split(images, num_threads)
    args = []
    for cnt, sub_images in enumerate(images):
        args.append((sub_images, os.path.join(resources.temp_chips, str(cnt).zfill(4)), shapefile, value_field,
                     tile_size, stride, rotation))
    pool = ThreadPool(create_training_chips, args, num_threads)
    pool.start()
    pool.join()
    merge_results(chips_dir)


# Transfer unique color values into your own binary label-chips
def unique_colors_to_masks(image, labels, dst_dir):
    name, extension = os.path.splitext(os.path.basename(image))
    if not os.path.exists(os.path.join(dst_dir, name)):
        os.makedirs(os.path.join(dst_dir, name))
    shutil.copy(image, os.path.join(dst_dir, name, os.path.basename(image)))
    for label in labels:
        name, extension = os.path.splitext(os.path.basename(label))
        roof = str(os.path.split(os.path.split(label)[0])[1]).zfill(4)
        if not os.path.exists(os.path.join(dst_dir, name, roof)):
            os.makedirs(os.path.join(dst_dir, name, roof))
        img = skimage.io.imread(label)
        if len(img.shape) == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        for i in range(1, len(colors)):  # skip 0 = background
            mask = (img == colors[i]).all(-1).reshape((512, 512, 1))
            with skimage.external.tifffile.TiffWriter(
                    os.path.join(dst_dir, name, roof, "{}.tif".format(str(i).zfill(9)))) as tif:
                tif.save((mask * 255).astype(np.uint8), compress=9)


def unique_colors_to_masks_mt(chips_dir, dst_dir, num_threads=None):
    with open(os.path.join(chips_dir, "map.txt"), "r") as file:
        lines = file.readlines()
    args = []
    for line in lines:
        split = line.split()
        image = os.path.join(chips_dir, split[0])
        labels = [os.path.join(chips_dir, label) for label in split[1:]]
        args.append((image, labels, dst_dir))
    pool = ThreadPool(unique_colors_to_masks, args, num_threads)
    pool.start()
    pool.join()
    write_aug_file(dst_dir)
    ic = ImageSetCollection(dst_dir)
    ic.write_augmented()
    ic.write_stats()

# Auxiliary functions for remembering already augmented data points
def write_aug_file(dst_dir):
    image_sets = [image_set for image_set in os.listdir(dst_dir) if os.path.isdir(os.path.join(dst_dir, image_set))]
    with open(os.path.join(dst_dir, "aug.txt"), "w+") as file:
        for image_set in image_sets:
            file.write("{}\t{}\n".format(image_set, False))

# Enrich the data with additional information (edge detection)
def enrich_rgb_channels(image_set, norm="uint8"):
    basename = os.path.basename(image_set)
    shutil.copy(os.path.join(image_set, "{}.tif".format(basename)), os.path.join(image_set, "{}_bak.tif".format(basename)))
    normalized = normalization.normalize(os.path.join(image_set, "{}.tif".format(basename)), norm)
    with skimage.external.tifffile.TiffWriter(os.path.join(image_set, "{}.tif".format(basename))) as tif:
        tif.save(normalized, compress=9)


def enrich_rgb_channels_mt(chips_dir, num_threads=None, norm="uint8"):
    image_sets = [(os.path.join(chips_dir, image_set), norm) for image_set in os.listdir(chips_dir) if os.path.isdir(os.path.join(chips_dir, image_set))]
    pool = ThreadPool(enrich_rgb_channels, image_sets, num_threads)
    pool.start()
    pool.join()