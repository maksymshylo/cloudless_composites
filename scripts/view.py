import datacube
from datacube.storage.masking import mask_invalid_data  
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scripts.combiner import create_composite, get_mask


def ploting_rgb_xarray_for_one_day(x_array,time_index, min_possible=0, max_possible=10000,min_inten=0.15, max_inten=1.0,bands = ['red', 'green', 'blue']):
    """
    Plots all datasets which contains x_array

    Args:
        x_array - loaded xarray from server
        time_index: index of needed time in xarray dataset

    Returns:
        rgb image of needed time
    """
    rgb = np.stack([x_array[bands[0]],
                    x_array[bands[1]],
                    x_array[bands[2]]], axis = -1)
    
    min_rgb =  min_possible
    max_rgb =  max_possible
    rgb = np.interp(rgb, (min_rgb, max_rgb), [min_inten,max_inten])
    rgb = rgb.astype(float)
    fake_saturation = 5000
    #rgb /= fake_saturation  # scale to [0, 1] range for imshow
    figure(figsize=(13,13))
    plt.imshow(rgb[time_index])
    
    

def plotting_xarray(data):
    """
    Args:
        x_array - loaded xarray from server

    Returns: all images in xarray
    """
    dc = datacube.Datacube(app='plot-rgb-recipe')
    fake_saturation = 4000
    rgb = data[['red','green','blue']].to_array(dim='color')
    rgb = rgb.transpose(*(rgb.dims[1:]+rgb.dims[:1]))  # make 'color' the last dimension
    rgb = rgb.where((rgb <= fake_saturation).all(dim='color'))  # mask out pixels where any band is 'saturated'
    rgb /= fake_saturation  # scale to [0, 1] range for imshow
    #print(rgb.shape)
    #print(rgb)
    rgb.plot.imshow(x=data.crs.dimensions[1], y=data.crs.dimensions[0],
                col='time', col_wrap=5, add_colorbar=False)
    

def plot_composite(composite,min_possible=0, max_possible=10000,min_inten=0.15, max_inten=1.0):
    """
    composite: - composite with bands RGBN, shape(4,x,y)

    """
    composite = composite[[0,1,2],:,:]
    rgb = np.moveaxis(composite, 0,2)
    min_rgb = min_possible
    max_rgb = max_possible
    rgb = np.interp(rgb, (min_rgb, max_rgb), [min_inten,max_inten])
    rgb = rgb.astype(float)
    figure(figsize=(13,13))
    return plt.imshow(rgb)
    
def apply_true_false_mask(array):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(~array, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')

def plot_comp_from_mask(xarray, day_idx, mask_class):
    '''
    xarray - input array(xarray)
    day_indx - number of day
    mask_class - class of mask
    '''
    print(xarray.slc.attrs)
    #exporting mask
    base_mask = get_mask(xarray,day_idx).astype('int16')
    #choosing needed class
    base_defects = (base_mask == mask_class)
    base_defects = base_defects[np.newaxis,:,:]
    
    #creating 4 copies of base_defects
    base_defects= np.vstack([np.copy(base_defects),
              np.copy(base_defects),
              np.copy(base_defects),
              np.copy(base_defects)])
    base_defects = np.moveaxis(base_defects,0,-1)
    
    comp = np.moveaxis(create_composite(xarray,day_idx),0,-1)
    #changing values
    comp = np.where(base_defects==False, comp, [10000,10000,5000,0])
    comp = np.moveaxis(comp,2,0)
    #ploting
    plot_composite(comp)