import skimage.io
import skimage.filters
import numpy as np

def normalize(image, norm="uint8", kernel_size=3):
    def normalize_to_uint16(image, steps_per_meter=3.): # No effect on training performance
        image *= steps_per_meter
        return image.astype(np.uint16)
    def normalize_to_uint8(image): # 8 bit sampling
        i_min = np.min(image)
        i_max = np.max(image)
        i_w = i_max - i_min
        img = np.copy(image)
        img -= i_min
        img *= 255. / i_w
        return img.astype(np.uint8)
    image = skimage.io.imread(image)
    image[image < 0.] = 0.
    image_sx = skimage.filters.sobel_h(image)
    image_sy = skimage.filters.sobel_v(image)
    if norm == "uint16":
        image = normalize_to_uint16(image)
        image_sx = normalize_to_uint16(image_sx)
        image_sy = normalize_to_uint16(image_sy)
    elif norm == "uint8_c":
        image = normalize_to_uint8(image)
        image_sx[image_sx != 0.] = 255
        image_sx = image_sx.astype(np.uint8)
        image_sy[image_sy != 0.] = 255
        image_sy = image_sy.astype(np.uint8)
    else:
        image = normalize_to_uint8(image)
        image_sx = normalize_to_uint8(image_sx)
        image_sy = normalize_to_uint8(image_sy)
 
    image = np.stack([image, image_sx, image_sy], axis=2)
    return image
