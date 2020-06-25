from datetime import datetime
from datacube.storage.masking import mask_invalid_data  
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from scripts.combiner import create_composite, get_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os

def drop_empty_data(data: xr.Dataset) -> xr.Dataset:
    
    '''
    Drops empty photos from `data`.
    
    Args:
        data(xarray.Dataset): Input raw data.
        
    Returns :
        xarray.Dataset: Processed `data` without empty files.
    '''
    
    drop_list = []
    
    for time in data.time.values:
        if np.isnan(data.sel(time=time)[['red','green','blue','nir']].to_array()).all():
            drop_list.append(time)
        
    return data.drop(drop_list, dim='time')

def drop_threshold_cloudiness(data: xr.Dataset, threshold: float = 0.35, save_dims: bool=False) -> xr.Dataset:
    
    '''
    Drops photos with cloudiness percentage > `threshold` from `data`.
    
    Args:
        data(xarray.Dataset): Input raw data.
        
        threshold(flat): Treshold percentage of cloudiness to drop a photo.
        
        save_dims(bool): Fill data with zeros instead of dropping it.
        
    Returns :
        xarray.Dataset: Processed data without cloudy photos.
    '''
    
    df = get_info(data)
    idx =df[df['cloudiness_percentage'] > threshold].index
    time_idx = data.time.values[idx]
    
    if not save_dims:
        return data.drop(time_idx, dim='time')
    else:
        for i in time_idx:
            data.sel(time=i).red.values[:] = 0
            data.sel(time=i).green.values[:] = 0
            data.sel(time=i).blue.values[:] = 0
            data.sel(time=i).nir.values[:] = 0
            data.sel(time=i).slc.values[:] = 0
            
            
        return data 


def get_info(data: xr.Dataset) -> pd.DataFrame:
        
    '''
    Get percentages of Cloudiness, Cirrus, and NoData in `data`.
    
    Args:
        data(xarray.Dataset): Input data.
        
    Returns :
        df(pd.DataFrame): Percentages of clouds, cirrus and nan values in `data`.
    '''
      
    cloudiness = []
    cirrus = []
    nan = []
    
    for time in data.time.values:
        mask = data.slc.sel(time=time).values
                
        data_size = np.sum(mask != 0)
        
        cloudiness_mask = (mask == 2)  | (mask == 3)  | (mask == 8)  | (mask == 9) | (mask == 10) 
        cirrus_mask = (mask == 10)
        
        count_cloudiness =  np.sum(cloudiness_mask == True)/data_size if data_size!=0 else 0
        count_cirrus =  np.sum(cirrus_mask == True)/data_size if data_size!=0 else 0
        count_nan =  np.sum(mask == 0)/(mask.shape[0]*mask.shape[1])
        
        cloudiness.append(count_cloudiness)
        cirrus.append(count_cirrus)
        nan.append(count_nan)
        
    df = pd.DataFrame({'time' : [str(i) for i in data.time.values],
                      'cloudiness_percentage': cloudiness,
                      'cirrus_percetage' : cirrus,
                      'nan_percetage of whole comp': nan})
    
    return df
   
    

def ch_mask(mask,i,j):
    """
    returns True if [i,j] pixel is cloudinesed, False if it is not
    """
    if (mask[i,j] == 10) | (mask[i,j] == 2)  | (mask[i,j] == 3)  |(mask[i,j] == 8)  | (mask[i,j] == 7)|(mask[i,j] == 9) |(mask[i,j] == 11):
        return True
    else:
        return False

def use_mask(x_array, time_index):
    """
    returns 4-bands numpy array which is composite whis used mask
    """
    mask = get_mask(data_xr,time_index)
    comp = create_composite(data_xr,time_index)
    new_comp = np.empty((comp.shape))
    for i in range(comp[0,:,:].shape[0]):
        for j in range(comp[0,:,:].shape[1]):
            if ch_mask(mask,i,j):
                new_comp[:,i,j] = 0
            else:
                new_comp[:,i,j] = comp[:,i,j]
                
    return new_comp

def save_from_narray_to_geotiff(x_array, array, path, pixel_size = 10):
    
    x_pixels = array.shape[2]  # number of pixels in x
    y_pixels = array.shape[1]  # number of pixels in y
    bands = array.shape[0]     # number of bands
    
    transform = (x_array.x.values[0], pixel_size, 0.0, x_array.y.values[0], 0.0, -pixel_size)
    
    new_dataset = rasterio.open(path, 'w',
                                driver ='GTiff',
                                height = y_pixels,
                                width  = x_pixels,
                                count  = bands,
                                dtype  = array.dtype,
                                crs    = x_array.crs.wkt,
                                transform =Affine.from_gdal(*transform))
    for i in range(bands):
        new_dataset.write(array[i,:,:], i+1)
    new_dataset.close()
'''
def save_from_narray_to_geotiff(x_array, array, path, pixel_size = 10):
    x_pixels = array.shape[2]  # number of pixels in x
    y_pixels = array.shape[1]  # number of pixels in y
    bands = array.shape[0]     # number of bands
    dst_crs = 'EPSG:4326'
    transform = (x_array.x.values[0], pixel_size, 0.0, x_array.y.values[0], 0.0, -pixel_size)
    path2 = path[0:-4]+'r.tif'
    new_dataset = rasterio.open(path2, 'w',
                                driver ='GTiff',
                                height = y_pixels,
                                width  = x_pixels,
                                count  = bands,
                                dtype  = array.dtype,
                                crs    = x_array.crs.wkt,
                                transform =Affine.from_gdal(*transform))
    for i in range(bands):
        new_dataset.write(array[i,:,:], i+1)
    transform, width, height = calculate_default_transform(new_dataset.crs, dst_crs, new_dataset.width, new_dataset.height, *new_dataset.bounds)
    kwargs = new_dataset.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    with rasterio.open(path, 'w', **kwargs) as dst:
        for i in range(1, new_dataset.count + 1):
            reproject(
                source=rasterio.band(new_dataset, i),
                destination=rasterio.band(dst, i),
                src_transform=new_dataset.transform,
                src_crs=new_dataset.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
    new_dataset.close()
    os.system('rm '+path2)
'''
class DataLoader():
    def __init__(self, dc, latitude, longitude,time, chunk_size  = (2000,2000), 
                platform = 'SENTINEL_2',product = 's2_l2a_kyiv_10m',
                measurements = ['red', 'green', 'blue', 'nir', 'slc'], view_with_nodata = False):
        
        self.dc = dc
        self.latitude = latitude
        self.longitude = longitude
        self.time = time 
        self.platform = platform
        self.measurements = measurements
        self.chunk_size = chunk_size
        self.view_with_nodata = view_with_nodata
        
        self.data = dc.load(latitude = latitude,
                            longitude = longitude,                         
                            platform = platform,
                            time = time,
                            product = product,
                            measurements = measurements,
                            dask_chunks = {"time":1, "x":chunk_size[0], "y":chunk_size[1]}
                            )
        
        self.indexes = self.split_indexes()
        
    def __len__(self):
        return len(self.indexes)
    
    
    def __getitem__(self, idx: int) -> xr.Dataset:
        x_range,y_range  =  self.indexes[idx]

        chunk = self.data[dict(x=slice(*x_range), y=slice(*y_range))] 
      
        # replace 99999 with NaN 
        chunk = mask_invalid_data(chunk)        
        # replace NaN in masks with 0 
        chunk.slc.values = np.nan_to_num(chunk.slc.values,nan =0, copy=True)\
                           .astype(np.uint8) 
       
        return chunk.compute() if self.load_data_ else chunk
        
        
    
    def __str__(self) ->str:
        return 'Data of the region: lat={},long={}, time={}.\n'\
               .format(self.latitude,self.longitude,self.time, )+\
               'Consists of {} chunks x {} x {} px each.\n'\
               .format(len(self), *self.chunk_size) +\
               'Matrix of chunks: \n{}'.format(self.matrix_of_chunks)
    
    def load_data(self, flag: bool) -> None:
        self.load_data_ = flag
              
    def split_indexes(self) ->list:
        x_pix = self.data.x.shape[0]
        y_pix = self.data.y.shape[0]
    
        x_idx =  list(range(0,x_pix , self.chunk_size[0]))
        y_idx =  list(range(0,y_pix , self.chunk_size[1]))
    
        x_idx.append(x_pix)
        y_idx.append(y_pix)
      
    
        c = []
        for j in range(len(y_idx)-1):
            for i in range(len(x_idx)-1):
                chunk_x = (x_idx[i], x_idx[i+1])
                chunk_y = (y_idx[j], y_idx[j+1])
                c.append([chunk_x,chunk_y])
                

    
                    
        self.matrix_of_chunks = np.array(list(range(len(c)))).reshape((len(y_idx)-1, len(x_idx)-1))
        
        if self.view_with_nodata:
            self.matrix_of_chunks = list(range(len(c)))
            for idx in range(len(c)):
                x_range,y_range = c[idx]
                chunk = self.data[dict(x=slice(*x_range), y=slice(*y_range))]
                self.matrix_of_chunks[idx] = -idx if  (chunk.slc.values == 0).all() else idx
            self.matrix_of_chunks = np.array(self.matrix_of_chunks).reshape((len(y_idx)-1, len(x_idx)-1))
        print("Matrix of chunks: ")
        print(self.matrix_of_chunks)
         
        return c