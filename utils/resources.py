import os

# ----------- DIRS -----------
root = os.getcwd()
images = os.path.join(os.getcwd(), "./resources/images/")
logs = os.path.join(os.getcwd(), "./resources/logs/")

images = os.path.join(os.getcwd(), "./resources/images/")
images_dom = os.path.join(os.getcwd(), "./resources/images/dom")
images_dgm = os.path.join(os.getcwd(), "./resources/images/dgm")
images_ndom = os.path.join(os.getcwd(), "./resources/images/ndom")
images_chips_esri = os.path.join(os.getcwd(), "./resources/images/chips/esri")
images_chips_single_masks = os.path.join(os.getcwd(), "./resources/images/chips/single_masks")
images_train_test_val = os.path.join(os.getcwd(), "./resources/images/chips/train_test_val")

laz = os.path.join(os.getcwd(), "./resources/temp/laz/")
las = os.path.join(os.getcwd(), "./resources/temp/las/")

temp = os.path.join(os.getcwd(), "./resources/temp/")
temp_centroids = os.path.join(os.getcwd(), "./resources/temp/centroids")
temp_extracted = os.path.join(os.getcwd(), "./resources/temp/extracted/")
temp_footprints = os.path.join(os.getcwd(), "./resources/temp/footprints/")
temp_split_ndom = os.path.join(os.getcwd(), "./resources/temp/split_ndom")
temp_rescaled = os.path.join(os.getcwd(), "./resources/temp/rescaled/")
temp_shifted = os.path.join(os.getcwd(), "./resources/temp/shifted/")
temp_vectorized = os.path.join(os.getcwd(), "./resources/temp/vectorized/")
temp_chips = os.path.join(os.getcwd(), "./resources/temp/chips")

# ----------- FILES -----------
rulefile = os.path.join(os.getcwd(), "./resources/rules.rpk")
backbone = os.path.join(os.getcwd(), "./resources/mask_rcnn_coco.h5")
building_gdb = os.path.join(os.getcwd(), "resources/buildings_3d.gdb")
shapefile = os.path.join(os.getcwd(), "./resources/shapes/AX_Gebaeude.shp")
laszip = os.path.join(os.getcwd(), "./resources/tools/laszip.exe")