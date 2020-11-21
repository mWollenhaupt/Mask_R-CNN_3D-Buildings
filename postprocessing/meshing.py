import os
import sys
import arcpy
import numpy as np
import skimage.io
ROOT_DIR = os.path.abspath("../")
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)
from utils import resources
from model import dataset
from utils.logger import ProgressLogger

def mesh(detections, splits, shapefile, dst_name="buildings_3d"):
    # georeferencing
    def shift_and_rescale(image, mask_arr):
        basename = os.path.splitext(os.path.basename(image))[0]
        image = arcpy.sa.Raster(image)
        mask = arcpy.NumPyArrayToRaster(mask_arr.astype(np.uint8) * 255)
        arcpy.env.outputCoordinateSystem = image
        os.makedirs(resources.temp_shifted, exist_ok=True)
        shifted = os.path.join(resources.temp_shifted, "{}.tif".format(basename))
        arcpy.Shift_management(mask,
                               shifted,
                               image.extent.XMin - mask.extent.XMin,
                               image.extent.YMin - mask.extent.YMin,
                               in_snap_raster=image
                               )
        # match LIDAR resolution
        os.makedirs(resources.temp_rescaled, exist_ok=True)
        rescaled = os.path.join(resources.temp_rescaled, "{}.tif".format(basename))
        arcpy.Rescale_management(shifted,
                                 rescaled,
                                 image.meanCellWidth,
                                 image.meanCellHeight
                                 )
        return rescaled

    # vertorize masks
    def vectorize(georef_mask):
        basename = os.path.splitext(os.path.basename(georef_mask))[0]
        os.makedirs(resources.temp_vectorized, exist_ok=True)
        vectorized = os.path.join(resources.temp_vectorized, "{}.shp".format(basename))
        field = "VALUE"
        arcpy.RasterToPolygon_conversion(georef_mask, vectorized, "NO_SIMPLIFY", field)
        return vectorized

    def get_centroid(polygon, mask):
        mask_ras = arcpy.sa.Raster(mask)
        centroid = None
        with arcpy.da.SearchCursor(polygon, ["SHAPE@", "gridcode"]) as cursor:
            for row in cursor:
                if row[1] == 255:
                    if mask_ras.extent.contains(row[0]):
                        centroid = row[0].centroid
        del cursor
        return centroid

    # work with quantiles to suppress outliers
    def pick_heights(image_arr, mask_arr):
        heights = (image_arr * mask_arr)
        heights = heights[np.nonzero(heights)]
        if len(heights) == 0:
            return None, None, None, None, None
        q_000 = np.quantile(heights, 0.0)
        q_010 = np.quantile(heights, 0.1)
        q_050 = np.quantile(heights, 0.5)
        q_090 = np.quantile(heights, 0.9)
        q_100 = np.quantile(heights, 1.0)
        return q_000, q_010, q_050, q_090, q_100

    def write_centroids_to_feature_class(centroids):
        os.makedirs(resources.temp_centroids, exist_ok=True)
        fc = arcpy.CreateFeatureclass_management(resources.temp_centroids, "centroids.shp", "POINT")
        fc = fc.getOutput(0)
        arcpy.AddField_management(fc, "Roof", "TEXT")
        arcpy.AddField_management(fc, "Score", "FLOAT")
        arcpy.AddField_management(fc, "Perc_000", "FLOAT")
        arcpy.AddField_management(fc, "Perc_010", "FLOAT")
        arcpy.AddField_management(fc, "Perc_050", "FLOAT")
        arcpy.AddField_management(fc, "Perc_090", "FLOAT")
        arcpy.AddField_management(fc, "Perc_100", "FLOAT")
        with arcpy.da.InsertCursor(fc, ["SHAPE@", "Roof", "Score", "Perc_000", "Perc_010", "Perc_050", "Perc_090",
                                        "Perc_100"]) as cursor:
            for centroid in centroids:
                cursor.insertRow(
                    [centroid[0], centroid[1], centroid[2], centroid[3], centroid[4], centroid[5], centroid[6],
                     centroid[7]])
        del cursor
        return fc

    def get_footprints(centroids, shapefile):
        os.makedirs(resources.temp_footprints, exist_ok=True)
        output_feature_class = os.path.join(resources.temp_footprints, "unsorted.shp")
        join_operation = "JOIN_ONE_TO_MANY"
        join_type = "KEEP_COMMON"
        match_option = "CONTAINS"
        arcpy.SpatialJoin_analysis(shapefile, centroids, output_feature_class,
                                   join_operation=join_operation,
                                   join_type=join_type,
                                   match_option=match_option)
        fields = [field.name for field in arcpy.ListFields(output_feature_class)]
        valids = ["FID", "Shape", "Roof", "Score", "Perc_000", "Perc_010", "Perc_050", "Perc_090", "Perc_100"]
        to_drop = [item for item in fields if item not in valids]
        arcpy.DeleteField_management(output_feature_class, to_drop)
        output = os.path.join(resources.temp_footprints, "footprints.shp")
        arcpy.Sort_management(output_feature_class, output, "Shape ASCENDING", "UL")
        arcpy.Delete_management(output_feature_class)
        arcpy.DeleteIdentical_management(output, "Shape", None, 0)
        return output

    # generate 3d model via CityEngine-Support
    def create_3d_model(footprints, rule_package, name):
        if not os.path.exists(resources.building_gdb):
            arcpy.CreateFileGDB_management(os.path.dirname(resources.building_gdb),
                                           os.path.basename(resources.building_gdb))
        arcpy.FeaturesFromCityEngineRules_3d(
            footprints,
            rule_package,
            os.path.join(resources.building_gdb, name))
        return os.path.join(resources.building_gdb, name)

    if type(splits) != list or type(detections) != list:
        raise Exception("Images and detections must be passed as a list!")
    if len(splits) != len(detections):
        raise Exception("LEN of images and detections must be equal!")

    centr_with_attr = []
    logger = ProgressLogger(len(splits), mesh)
    for i in range(len(splits)):
        image = splits[i]
        image_arr = skimage.io.imread(image)
        detection = detections[i]
        masks = detection["masks"]
        if len(detection["class_ids"]) == 0:
            logger.log_progress()
            continue
        for j in range(masks.shape[2]):
            mask_arr = masks[:, :, j]
            georef_mask = shift_and_rescale(image, mask_arr)
            vectorized = vectorize(georef_mask)
            centroid = get_centroid(vectorized, georef_mask)
            q_000, q_010, q_050, q_090, q_100 = pick_heights(image_arr, mask_arr)
            if centroid is None or q_000 is None:
                continue
            score = float(detection["scores"][j])
            roof = dataset.roofs[detection["class_ids"][j]]
            centr_with_attr.append([centroid, roof, score, q_000, q_010, q_050, q_090, q_100])
        logger.log_progress()
    print("start writing centroids to feature class...")
    centroid_features = write_centroids_to_feature_class(centr_with_attr)
    print("done.")
    print("start picking footprints..")
    footprints = get_footprints(centroid_features, shapefile)
    print("done.")
    print("start meshing..")
    model = create_3d_model(footprints, resources.rulefile, dst_name)
    print("done.")
    return model