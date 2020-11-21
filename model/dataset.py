from matterport import utils
from utils.multiprocessing import ThreadPool
import os
import random
import shutil
import numpy as np
import skimage.io

# label mapping
roofs = {1: "FD",
         2: "PD",
         3: "SD",
         4: "WD",
         5: "KWD",
         6: "MD"}

class RoofTypeDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        self.roofs = []
        self.types = { # mapping of ALKIS-keys
            "1000":1,
            "2100":2,
            "3100":3,
            "3200":4,
            "3300":5,
            "3400":6
        }

    def load_data(self, dataset_dir):
        # add classes
        self.add_class("roof_types", 1, "1000")
        self.add_class("roof_types", 2, "2100")
        self.add_class("roof_types", 3, "3100")
        self.add_class("roof_types", 4, "3200")
        self.add_class("roof_types", 5, "3300")
        self.add_class("roof_types", 6, "3400")

        # add images
        self.dataset_dir = dataset_dir
        for image_set in os.listdir(self.dataset_dir):
            image = os.path.join(self.dataset_dir, image_set, '{}.tif'.format(image_set))
            self.add_image(
                "roof_types",
                image_id=os.path.split(image)[0],
                path=image
            )

    def load_mask(self, image_id):
        """Load instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "roof_types":
            return super(self.__class__, self).load_mask(image_id)

        mask_arrs = []
        mask_lbls = np.empty(0).astype(np.int)
        image_id = image_info['id']
        mask_dirs = [os.path.join(image_id, md) for md in os.listdir(image_id) if
                     os.path.isdir(os.path.join(image_id, md))]
        for mask_dir in mask_dirs:
            mask_type = os.path.split(mask_dir)[1]
            for file in os.listdir(mask_dir):
                arr = skimage.io.imread(os.path.join(mask_dir, file)).astype(np.bool)
                mask_arrs.append(arr)
                mask_lbls = np.append(mask_lbls, self.types[mask_type])
        mask_stack = np.dstack(np.asarray(mask_arrs))
        return mask_stack, mask_lbls

    def image_reference(self, image_id): # for debugging only
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "roof_types":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
# "try and error"-splitting            
def split_train_test_val_mt(src, dst, _train=0.8, _test=0.1, _val=0.1, num_threads=None):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst, "test"), exist_ok=True)
    os.makedirs(os.path.join(dst, "val"), exist_ok=True)
    folders = os.listdir(src)
    idx = list(range(len(folders)))
    random.Random(101010).shuffle(idx)
    train_idx = int(_train*len(idx))
    val_idx = train_idx+(int(_val*len(idx)))
    train_dirs = idx[0:train_idx]
    val_dirs = idx[train_idx:val_idx]
    test_dirs = idx[val_idx:]
    args = []
    for i in range(len(train_dirs)):
        args.append((os.path.join(src, folders[train_dirs[i]]), os.path.join(dst, "train", folders[train_dirs[i]])))
    for i in range(len(val_dirs)):
        args.append((os.path.join(src, folders[val_dirs[i]]), os.path.join(dst, "val", folders[val_dirs[i]])))
    for i in range(len(test_dirs)):
        args.append((os.path.join(src, folders[test_dirs[i]]), os.path.join(dst, "test", folders[test_dirs[i]])))
    pool = ThreadPool(shutil.move, args, num_threads)
    pool.start()
    pool.join()