import os
import torch
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import colors
import seaborn as sns
import numpy as np
from shapely.geometry import mapping, box
import rasterio as rio
from rasterio.plot import plotting_extent
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
from rasterio import mask
from rasterio.enums import Resampling
import numpy.ma as ma

# read the tiff and give the field values

device = torch.device('cuda', 1)

# Useful functions


def reverse_coordinates(pol):
    """
    Reverse the coordinates in pol
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    Returns [[y1,x1],[y2,x2],...,[yN,xN]]
    """
    return [list(f[-1::-1]) for f in pol]


def to_index(wind_):
    """
    Generates a list of index (row,col): [[row1,col1],[row2,col2],[row3,col3],[row4,col4],[row1,col1]]
    """
    return [[wind_.row_off, wind_.col_off],
            [wind_.row_off, wind_.col_off + wind_.width],
            [wind_.row_off + wind_.height, wind_.col_off + wind_.width],
            [wind_.row_off + wind_.height, wind_.col_off],
            [wind_.row_off, wind_.col_off]]


def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]],
            [bbox[0], bbox[3]], [bbox[0], bbox[1]]]


def pol_to_np(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    return np.array([list(l) for l in pol])


def pol_to_bounding_box(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    arr = pol_to_np(pol)
    return BoundingBox(np.min(arr[:, 0]), np.min(arr[:, 1]), np.max(arr[:, 0]),
                       np.max(arr[:, 1]))


def setTheMainDevice(main_device):
    device = main_device


def shpExtractVertices(shp_path):
    """This will read the shp file and return all the rectangular vertices in the file
    
    Arguments:
        shp_path {[str]} -- [the path for the shp file]
    
    Returns:
        [list] -- [will be a 3D array list]
    """
    df = gpd.read_file(shp_path)
    g = [i for i in df.geometry]

    all_coords = []
    for b in g[0].boundary:  # for first feature/row
        coords = np.dstack(b.coords.xy).tolist()
        all_coords.append(*coords)
    return all_coords


def degreeToMeter(degreeUnit):
    meter = degreeUnit * (2. * np.math.pi * 6371004.) / 360.
    return meter


def read_and_resampleRaster(rasterPath, demMeta, device):
    with rio.open(rasterPath) as dataset:
        # resample data to target shape
        upscale_factor = dataset.meta['transform'][0] / demMeta['transform'][0]
        data = dataset.read(1,
                            out_shape=(dataset.count,
                                       int(dataset.height * upscale_factor),
                                       int(dataset.width * upscale_factor)),
                            resampling=Resampling.bilinear)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-2]),
            (dataset.height / data.shape[-1]))

        rows_min, cols_min = rio.transform.rowcol(transform,
                                                  demMeta['transform'][2],
                                                  demMeta['transform'][5])

        resampleData = data[rows_min:rows_min + demMeta['width'],
                            cols_min:cols_min + demMeta['height']]
        resampleData = torch.from_numpy(resampleData).to(device=device)

        return resampleData


# ===========================================================================
# GPU version, will cause problems when big raster
# ===========================================================================
def importDEMData(DEM_path, device):
    with rio.open(DEM_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    dem = np.ma.filled(demMasked, fill_value=-9999.)
    mask = demMasked.mask

    dem = torch.from_numpy(dem).to(device=device)
    mask = torch.from_numpy(mask).to(device=device)

    # dem_min = torch.zeros_like(dem)
    # dem_min += 10000.

    maskID = mask.to(torch.int32)
    # mask = ~mask

    oppo_direction = np.array([[-1, 1], [1, 0], [1, 1], [-1, 0]])
    # maskMatrix = torch.
    mask_boundary = torch.zeros_like(mask, dtype=torch.int32, device=device)
    for i in range(4):
        # dem_min = torch.min(
        #     dem_min,
        #     dem.roll(int(oppo_direction[i][0]),
        #              int(oppo_direction[i][1])).abs())
        mask_boundary = mask_boundary + maskID.roll(int(oppo_direction[i][0]),
                                                    int(oppo_direction[i][1]))

    # here we get the dem after considering fill sink
    # dem = torch.where(dem > -1000., torch.max(dem, dem_min), dem)

    mask_boundary[mask] = 0

    mask_boundary = mask_boundary.to(torch.bool)

    mask = ~mask

    return dem, mask, mask_boundary, demMeta


# enum BOUNDARY_TYPE = {WALL_NON_SLIP = 3, WALL_SLIP = 4, OPEN = 5, HQ_GIVEN =6};
def setBoundaryEdge(mask,
                    mask_boundary,
                    demMeta,
                    device,
                    boundBox=np.array([]),
                    bc_type=6):
    # boundBox = [[x_min, y_min, x_max, y_max],[x_min, y_min, x_max, y_max]]
    # it will be great if you can get the box from gis tools
    mask = mask.to(torch.int32)
    bc_dict = {'WALL_NON_SLIP': 3, 'WALL_SLIP': 4, 'OPEN': 5, 'HQ_GIVEN': 6}
    # set the default BC as HQ_GIVEN, and H and Q are 0.0

    mask[mask_boundary] = 60
    if boundBox.size > 0:
        mask_box = torch.zeros_like(mask, dtype=torch.bool, device=device)
        for i in range(len(boundBox)):
            if type(bc_type[i]) == str:
                try:
                    BC_TYPE = bc_dict[bc_type[i]]
                except KeyError:
                    print(
                        "The keys should be: WALL_NON_SLIP, WALL_SLIP, OPEN, HQ_GIVEN"
                    )
            else:
                BC_TYPE = bc_type[i]
            rows_min, cols_min = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][0],
                                                      boundBox[i][1])
            rows_max, cols_max = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][2],
                                                      boundBox[i][3])
            mask_box[rows_min:rows_max, cols_min:cols_max] = True

            mask[mask_boundary & mask_box] = int(str(BC_TYPE) + str(i))
    return mask


# ===========================================================================
# CPU version, no limitation for raster size
# ===========================================================================
def importDEMData_And_BC(DEM_path,
                         device,
                         gauges_position=np.array([]),
                         boundBox=np.array([]),
                         bc_type=6):
    with rio.open(DEM_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    dem = np.ma.filled(demMasked, fill_value=-9999.)
    mask = demMasked.mask

    dem = torch.from_numpy(dem).to(device=device)
    # mask = torch.from_numpy(mask).to(device=device)
    mask = torch.from_numpy(mask)
    maskID = mask.to(torch.int32)

    oppo_direction = np.array([[-1, 1], [1, 0], [1, 1], [-1, 0]])

    mask_boundary = torch.zeros_like(mask, dtype=torch.int32)
    for i in range(4):
        mask_boundary = mask_boundary + maskID.roll(int(oppo_direction[i][0]),
                                                    int(oppo_direction[i][1]))

    mask_boundary[mask] = 0

    mask_boundary = mask_boundary.to(torch.bool)

    mask = ~mask

    mask = mask.to(torch.int32)

    gauge_index_1D = torch.tensor([])

    if gauges_position.size > 0:
        # mask_gauge = torch.tensor(mask)  # here make a copy of mask values
        mask_gauge = mask.clone()
        rows, cols = rio.transform.rowcol(demMeta['transform'],
                                          gauges_position[:, 0],
                                          gauges_position[:, 1])
        mask_gauge[rows, cols] = 100

        gauge_index_1D = torch.flatten(
            (mask_gauge[mask_gauge > 0] >= 99).nonzero()).type(torch.int64)

        # reorder the gauge_index
        rows = np.array(rows)
        cols = np.array(cols)
        array = rows * dem.size()[1] + cols
        order = array.argsort()
        ranks = order.argsort()

        gauge_index_1D = gauge_index_1D[ranks]

    bc_dict = {'WALL_NON_SLIP': 3, 'WALL_SLIP': 4, 'OPEN': 5, 'HQ_GIVEN': 6}
    # set the default BC as HQ_GIVEN, and H and Q are 0.0

    mask[mask_boundary] = 60

    if boundBox.size > 0:
        mask_box = torch.zeros_like(mask, dtype=torch.bool)
        for i in range(len(boundBox)):
            mask_box[:] = 0
            if type(bc_type[i]) == str:
                try:
                    BC_TYPE = bc_dict[bc_type[i]]
                except KeyError:
                    print(
                        "The keys should be: WALL_NON_SLIP, WALL_SLIP, OPEN, HQ_GIVEN"
                    )
            else:
                BC_TYPE = bc_type[i]
            rows_min, cols_min = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][0],
                                                      boundBox[i][1])
            rows_max, cols_max = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][2],
                                                      boundBox[i][3])
            mask_box[rows_min:rows_max, cols_min:cols_max] = True

            mask[mask_boundary & mask_box] = int(str(BC_TYPE) + str(i))

    mask_GPU = mask.to(device=device)
    del mask
    torch.cuda.empty_cache()
    return dem, mask_GPU, demMeta, gauge_index_1D


# ===========================================================================
# CPU version, no limitation for raster size, return value include coordinate info
# ===========================================================================
def importDEMData_And_BC_storm(DEM_path,
                               device,
                               gauges_position=np.array([]),
                               boundBox=np.array([]),
                               bc_type=6):
    with rio.open(DEM_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    dem = np.ma.filled(demMasked, fill_value=-9999.)
    mask = demMasked.mask

    dem = torch.from_numpy(dem).to(device=device)
    # mask = torch.from_numpy(mask).to(device=device)
    mask = torch.from_numpy(mask)
    maskID = mask.to(torch.int32)

    transform = demMeta['transform']
    nx, ny = demMeta['width'], demMeta['height']
    x, y = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5) * transform

    x = torch.from_numpy(x).to(device=device)
    y = torch.from_numpy(y).to(device=device)

    oppo_direction = np.array([[-1, 1], [1, 0], [1, 1], [-1, 0]])

    mask_boundary = torch.zeros_like(mask, dtype=torch.int32)
    for i in range(4):
        mask_boundary = mask_boundary + maskID.roll(int(oppo_direction[i][0]),
                                                    int(oppo_direction[i][1]))

    mask_boundary[mask] = 0

    mask_boundary = mask_boundary.to(torch.bool)

    mask = ~mask

    mask = mask.to(torch.int32)

    gauge_index_1D = torch.tensor([])

    if gauges_position.size > 0:
        # mask_gauge = torch.tensor(mask)  # here make a copy of mask values
        mask_gauge = mask.clone()
        rows, cols = rio.transform.rowcol(demMeta['transform'],
                                          gauges_position[:, 0],
                                          gauges_position[:, 1])
        mask_gauge[rows, cols] = 100

        gauge_index_1D = torch.flatten(
            (mask_gauge[mask_gauge > 0] >= 99).nonzero()).type(torch.int64)

        # reorder the gauge_index
        rows = np.array(rows)
        cols = np.array(cols)
        array = rows * dem.size()[1] + cols
        order = array.argsort()
        ranks = order.argsort()

        gauge_index_1D = gauge_index_1D[ranks]

    bc_dict = {'WALL_NON_SLIP': 3, 'WALL_SLIP': 4, 'OPEN': 5, 'HQ_GIVEN': 6}
    # set the default BC as HQ_GIVEN, and H and Q are 0.0

    mask[mask_boundary] = 60

    if boundBox.size > 0:
        mask_box = torch.zeros_like(mask, dtype=torch.bool)
        for i in range(len(boundBox)):
            mask_box[:] = 0
            if type(bc_type[i]) == str:
                try:
                    BC_TYPE = bc_dict[bc_type[i]]
                except KeyError:
                    print(
                        "The keys should be: WALL_NON_SLIP, WALL_SLIP, OPEN, HQ_GIVEN"
                    )
            else:
                BC_TYPE = bc_type[i]
            rows_min, cols_min = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][0],
                                                      boundBox[i][1])
            rows_max, cols_max = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][2],
                                                      boundBox[i][3])
            mask_box[rows_min:rows_max, cols_min:cols_max] = True

            mask[mask_boundary & mask_box] = int(str(BC_TYPE) + str(i))

    mask_GPU = mask.to(device=device)

    x_1d = x[mask_GPU]
    y_1d = y[mask_GPU]

    del mask
    torch.cuda.empty_cache()
    return dem, mask_GPU, demMeta, gauge_index_1D, x_1d, y_1d


# ===========================================================================
# CPU version, no limitation for raster size, return mask id for Discrete Element Model
# ===========================================================================
def importDEMData_And_BC_DEM(DEM_path,
                         device,
                         gauges_position=np.array([]),
                         boundBox=np.array([]),
                         bc_type=6):
    with rio.open(DEM_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    dem = np.ma.filled(demMasked, fill_value=-9999.)
    mask = demMasked.mask

    dem = torch.from_numpy(dem).to(device=device)
    # mask = torch.from_numpy(mask).to(device=device)
    mask = torch.from_numpy(mask)
    maskID = mask.to(torch.int32)

    oppo_direction = np.array([[-1, 1], [1, 0], [1, 1], [-1, 0]])

    mask_boundary = torch.zeros_like(mask, dtype=torch.int32)
    for i in range(4):
        mask_boundary = mask_boundary + maskID.roll(int(oppo_direction[i][0]),
                                                    int(oppo_direction[i][1]))
    mask_boundary[mask] = 0

    mask_boundary = mask_boundary.to(torch.bool)

    mask = ~mask

    mask = mask.to(torch.int32)

    gauge_index_1D = torch.tensor([])

    if gauges_position.size > 0:
        # mask_gauge = torch.tensor(mask)  # here make a copy of mask values
        mask_gauge = mask.clone()
        rows, cols = rio.transform.rowcol(demMeta['transform'],
                                          gauges_position[:, 0],
                                          gauges_position[:, 1])
        mask_gauge[rows, cols] = 100

        gauge_index_1D = torch.flatten(
            (mask_gauge[mask_gauge > 0] >= 99).nonzero()).type(torch.int64)

        # reorder the gauge_index
        rows = np.array(rows)
        cols = np.array(cols)
        array = rows * dem.size()[1] + cols
        order = array.argsort()
        ranks = order.argsort()

        gauge_index_1D = gauge_index_1D[ranks]

    bc_dict = {'WALL_NON_SLIP': 3, 'WALL_SLIP': 4, 'OPEN': 5, 'HQ_GIVEN': 6}
    # set the default BC as HQ_GIVEN, and H and Q are 0.0

    mask[mask_boundary] = 60

    if boundBox.size > 0:
        mask_box = torch.zeros_like(mask, dtype=torch.bool)
        for i in range(len(boundBox)):
            mask_box[:] = 0
            if type(bc_type[i]) == str:
                try:
                    BC_TYPE = bc_dict[bc_type[i]]
                except KeyError:
                    print(
                        "The keys should be: WALL_NON_SLIP, WALL_SLIP, OPEN, HQ_GIVEN"
                    )
            else:
                BC_TYPE = bc_type[i]
            rows_min, cols_min = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][0],
                                                      boundBox[i][1])
            rows_max, cols_max = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][2],
                                                      boundBox[i][3])
            mask_box[rows_min:rows_max, cols_min:cols_max] = True

            mask[mask_boundary & mask_box] = int(str(BC_TYPE) + str(i))

    mask_GPU = mask.to(device=device)

    ############################# x,y and nan_mask ####################
    dx = demMeta['transform'][0]
    row = torch.arange(mask.size()[0])
    col = torch.arange(mask.size()[1])
    xllcor = demMeta['transform'][2]
    yllcor = demMeta['transform'][5]
    y, x = torch.meshgrid(row, col)
    x = x.to(device=device, dtype=torch.float64)
    y = y.to(device=device, dtype=torch.float64)
    x = xllcor + (x + 0.5) * dx
    y = yllcor - (y + 0.5) * dx
 
    mask_bound_extent = mask
    for j in range(3):
        for i in range(4):
            mask_bound_extent = mask_bound_extent + mask_bound_extent.roll(int(oppo_direction[i][0]),
                                                        int(oppo_direction[i][1]))
    mask_bound_extent[mask>0] = 0
    mask_bound_extent_GPU = mask_bound_extent.to(device=device, dtype=torch.bool)

    mask_boundary_GPU = mask_boundary.to(device=device)
    
    x_GPU = x.to(device=device)
    y_GPU = y.to(device=device)
    del mask, mask_boundary, mask_bound_extent
    torch.cuda.empty_cache()
    return dem, x_GPU, y_GPU, mask_GPU, mask_boundary_GPU, mask_bound_extent_GPU, demMeta, gauge_index_1D


def importFloatingDebrisData(Debris_path, xllcor, yllcor, device):
    debrisData = PlyData.read('Debris_path')
    debris_x = debrisData.x - xllcor
    debris_y = debrisData.y - yllcor
    debris_z = debrisData.z
    


# ===========================================================================
# read the landuse and return the parameters
# ===========================================================================
def importLanduseData(LandUse_path, device, level=1):
    with rio.open(LandUse_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    landuse = np.ma.filled(demMasked, fill_value=-127)
    # landuse = demMashed
    mask = demMasked.mask

    if level == 1:
        landuse = (landuse / 10).astype(int) - 1
        landuse_index_class = len(np.unique(landuse)) - 1
        landuse = torch.from_numpy(landuse).to(device=device)
    else:
        landuse_index_class = len(np.unique(landuse)) - 1
        indexes = np.unique(landuse)
        landuse = torch.from_numpy(landuse).to(device=device)
        for i in range(1, len(indexes), 1):
            landuse[landuse == indexes[i]] = i - 1

    landuse_index = np.arange(landuse_index_class)

    return landuse, landuse_index


def importRainStationMask(rainmask_path, device):
    with rio.open(rainmask_path) as src:
        demMashed = src.read(1, masked=True)
        demMeta = src.meta
    rainmask = np.ma.filled(demMashed, fill_value=0)
    rainmask = rainmask.astype(np.int16)
    rainmask = torch.from_numpy(rainmask).to(device=device)
    mask = demMashed.mask
    return rainmask


def lidar_rainfall(lidarFolderPath):
    # time should be the time.tif
    lidarFiles = glob(lidarFolderPath + '*.tif')
    return lidarFiles


def voronoiDiagramGauge_rainfall_source(rainfall_matrix_path):
    rainSource = np.genfromtxt(rainfall_matrix_path)
    return rainSource


def importAscDEM(asc_DEM_path, DEM_path):
    DEM_Data = np.genfromtxt(asc_DEM_path, skip_header=6)
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)

    print(demMeta)

    mask = ~mask
    mask_cpu = mask.cpu().numpy()
    DEM_Data = ma.masked_array(DEM_Data, mask=mask_cpu)

    nodatavalue = -9999.
    DEM_Data = ma.filled(DEM_Data, fill_value=nodatavalue)

    DATA_meta = demMeta.copy()
    DATA_meta.update({'nodata': nodatavalue})
    topAddress = asc_DEM_path[:asc_DEM_path.rfind('.')]

    outPutName = topAddress + 'tif'
    # print(outPutName)
    with rio.open(outPutName, 'w', **DATA_meta) as outf:
        outf.write(data_cpu, 1)


def gaussian(x, y, x_mu, y_mu, sig):
    return np.exp(-(np.power(x - x_mu, 2.) + np.power(y - y_mu, 2.)) /
                  (2 * np.power(sig, 2.)))


def normalDistMask(DEM_path, device):
    with rio.open(DEM_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    cantho_mask = demMasked.mask

    rain_mask = cantho_mask.copy()
    cantho_mask = ~cantho_mask

    y = np.arange(cantho_mask.shape[0])
    x = np.arange(cantho_mask.shape[1])

    xv, yv = np.meshgrid(x, y)

    cantho_mask = cantho_mask * gaussian(
        xv, yv, cantho_mask.shape[1], cantho_mask.shape[0],
        (cantho_mask.shape[1] + cantho_mask.shape[0]) / 3.)

    cantho_mask = ma.masked_array(cantho_mask, mask=rain_mask)
    nodatavalue = -9999.
    cantho_mask = ma.filled(cantho_mask, fill_value=nodatavalue)
    DATA_meta = demMeta.copy()
    DATA_meta.update({'nodata': nodatavalue})
    DATA_meta.update({'dtype': np.float64})

    export = False
    if export:
        topAddress = DEM_path[:DEM_path.rfind('/')]
        outPutName = topAddress + "/rainMask.tif"
        with rio.open(outPutName, 'w', **DATA_meta) as outf:
            outf.write(cantho_mask, 1)
    rain_mask_tensor = torch.as_tensor(cantho_mask[cantho_mask > 0.])
    rain_mask_tensor.to(device)
    print(rain_mask_tensor.size())
    return rain_mask_tensor


def importStorm(stormTimeStep, filePath):
    data = np.genfromtxt(filePath, skip_header=1)
    # x = data[:, 7]
    # y = data[:, 8]
    # p = data[:, 9]
    # r = data[:, 10]
    # v = data[:, 11]
    t = np.arange(data.shape[0]) * stormTimeStep.reshape((-1, 1))
    stormDataArray = data[:, 5:]
    stormDataArray = np.column_stack(t, stormDataArray)
    return stormDataArray


if __name__ == "__main__":
    # landuse, landindex = importLanduseData(
    #     '/home/cvjz3/Luanhe_case/Luan_Data_90m/Landuse.tif', 2)
    # print(landuse[landuse >= 0])
    # print(landindex)
    # importAscDEM('/home/cvjz3/Luanhe_case/dem.asc',
    #              '/home/cvjz3/Luanhe_case/Luan_Data_90m/DEM.tif')
    normalDistMask('/home/cvjz3/CanTho/dem.tif')