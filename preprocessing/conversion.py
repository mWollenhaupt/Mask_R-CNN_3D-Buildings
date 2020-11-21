import os
import arcpy
import subprocess
from utils.multiprocessing import ThreadPool
from utils.logger import ProgressLogger
import shutil
from utils import resources

arcpy.env.overwriteOutput = True


def convert_LAZ_to_LAS(file, dst_dir, target_type):
    file_name, extension = os.path.splitext(os.path.basename(file))
    dst_dir = os.path.join(dst_dir, target_type)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    cmd = [resources.laszip]
    cmd.extend(("-i", file))
    cmd.extend(("-o", os.path.join(dst_dir, "{}_{}.las".format(file_name, target_type))))
    if target_type == 'dom':
        cmd.append('-first_only')
        cmd.extend(('-keep_classification', '1'))  # unclassified points
        cmd.extend(('-keep_classification', '2'))  # surface/ground points
        cmd.extend(('-keep_classification', '9'))  # synthetic water points
        cmd.extend(('-keep_classification', '17'))  # points on bridges
        cmd.extend(('-keep_classification', '20'))  # last returning non ground points
    if target_type == 'dgm':
        cmd.append('-last_only')
        cmd.extend(('-keep_classification', '2'))  # surface/ground points
        cmd.extend(('-keep_classification', '9'))  # synthetic water points
        cmd.extend(('-keep_classification', '17'))  # points on bridges
        cmd.extend(('-keep_classification', '20'))  # last returning non ground points
        cmd.extend(('-keep_classification', '21'))  # synthetic building points
        cmd.extend(('-keep_classification', '24'))  # basement points
        cmd.extend(('-keep_classification', '26'))  # synthetic ground points
    subprocess.call(cmd)

def convert_LAZ_to_LAS_mt(src_dir, dst_dir, num_threads=None):
    laz_files = [os.path.join(src_dir, file) for file in os.listdir(src_dir) if
                 os.path.isfile(os.path.join(src_dir, file))]
    args = []
    for file in laz_files:
        args.append((file, dst_dir, "dgm"))
        args.append((file, dst_dir, "dom"))
    pool = ThreadPool(convert_LAZ_to_LAS, args, num_threads)
    pool.start()
    pool.join()

# re-interpolate rasterized null-values
def check_nulldata(path):
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".tif")]
    logger = ProgressLogger(len(files), check_nulldata)
    for file in files:
        name = os.path.splitext(os.path.basename(file))[0]
        img = arcpy.sa.Raster(file)
        isn = arcpy.sa.IsNull(img)
        rect = 25
        i = 0
        while isn.minimum != isn.maximum:
            os.makedirs(os.path.join(resources.temp, "{}_{}".format(name, i)), exist_ok=True)
            nbr = (rect, rect)
            img = arcpy.sa.Con(arcpy.sa.IsNull(img),
                               arcpy.sa.FocalStatistics(img, arcpy.sa.NbrRectangle(nbr[0], nbr[1], "CELL"), "MEAN"),
                               img)
            cpy = arcpy.CopyRaster_management(img, os.path.join(resources.temp, "{}_{}".format(name, i),
                                                                "{}.tif".format(name)))
            del img
            cpy = cpy.getOutput(0)
            rect += 25
            img = arcpy.sa.Raster(cpy)
            isn = arcpy.sa.IsNull(img)
            if isn.minimum == isn.maximum:
                shutil.copy(os.path.join(resources.temp, "{}_{}".format(name, i), "{}.tif".format(name)), file)
                del img
                i += 1
                for j in range(i):
                    shutil.rmtree(os.path.join(resources.temp, "{}_{}".format(name, j)))
            i += 1
        logger.log_progress()


def convert_LAS_to_TIFF(src, dst, sampling_value=(1000 / 2048)):
    result = arcpy.LasDatasetToRaster_conversion(in_las_dataset=src,
                                        out_raster=dst,
                                        value_field="ELEVATION",
                                        interpolation_type="BINNING MAXIMUM LINEAR",
                                        data_type="FLOAT",
                                        sampling_value=sampling_value)

def convert_LAS_to_TIFF_mt(src_dir, dst_dir, num_threads=None):
    args = []
    las_dgm_files = [os.path.join(src_dir, "dgm", file) for file in os.listdir(os.path.join(src_dir, "dgm")) if
                 os.path.isfile(os.path.join(src_dir, "dgm", file))]
    dgm_dir = os.path.join(dst_dir, "dgm")
    if not os.path.exists(dgm_dir):
        os.makedirs(dgm_dir)
    for file in las_dgm_files:
        file_name, extension = os.path.splitext(os.path.basename(file))
        args.append((file, os.path.join(dgm_dir, "{}.tif".format(file_name))))
    las_dom_files = [os.path.join(src_dir, "dom", file) for file in os.listdir(os.path.join(src_dir, "dom")) if
                 os.path.isfile(os.path.join(src_dir, "dom", file))]
    dom_dir = os.path.join(dst_dir, "dom")
    if not os.path.exists(dom_dir):
        os.makedirs(dom_dir)
    for file in las_dom_files:
        file_name, extension = os.path.splitext(os.path.basename(file))
        args.append((file, os.path.join(dom_dir, "{}.tif".format(file_name))))
    pool = ThreadPool(convert_LAS_to_TIFF, args, num_threads)
    pool.start()
    pool.join()


def calculate_ndom(src_dom, src_dgm, dst):
    dom = arcpy.sa.Raster(src_dom)
    dgm = arcpy.sa.Raster(src_dgm)
    ndom = ((dom-dgm)>0)*(dom-dgm)
    name, extension = os.path.splitext(os.path.basename(src_dom))
    arcpy.CopyRaster_management(ndom, os.path.join(dst, "{}_ndom.tif".format(name[:-4])))


def calculate_ndom_mt(src_dom, src_dgm, dst, num_threads=None):
    os.makedirs(dst, exist_ok=True)
    doms = [os.path.join(src_dom, file) for file in os.listdir(src_dom) if file.endswith(".tif")]
    dgms = [os.path.join(src_dgm, file) for file in os.listdir(src_dgm) if file.endswith(".tif")]
    args = []
    for i in range(len(doms)):
        args.append((doms[i], dgms[i], dst))
    pool = ThreadPool(calculate_ndom, args, num_threads)
    pool.start()
    pool.join()

def split_ndom(img_path, delete=False):
    in_raster = arcpy.sa.Raster(img_path)
    if delete:
        if os.path.exists(resources.temp_split_ndom):
            shutil.rmtree(resources.temp_split_ndom)
    basename = "{}_".format(os.path.splitext(os.path.basename(img_path))[0])
    os.makedirs(os.path.join(resources.temp_split_ndom, basename), exist_ok=True)
    arcpy.SplitRaster_management(in_raster, os.path.join(resources.temp_split_ndom, basename), basename,
                                 "SIZE_OF_TILE", tile_size="512 512", overlap=256)
    imgs = [os.path.join(resources.temp_split_ndom, basename, img) for img in os.listdir(os.path.join(resources.temp_split_ndom, basename)) if img.lower().endswith(".tif")]
    return imgs