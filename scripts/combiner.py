import numpy as np
from scipy.ndimage import rank_filter
from skimage.morphology import square,dilation
import xarray as xr
import time
from tqdm import tqdm_notebook as tqdm
from typing import Callable
from sys import stdout


def create_composite(x_array, time_index, bands = ['red', 'green', 'blue','nir']):
    '''
    x_array: - dataset
    time_index:  - index of needed day
    '''
    one_day=x_array.sel(time=x_array.time.values[time_index])
    
    rgbn = np.stack([one_day[bands[0]],
                     one_day[bands[1]],
                     one_day[bands[2]],
                     one_day[bands[3]]], axis = -1)    
    return np.moveaxis(rgbn, 2, 0) # moveaxis для того щоб створити масив (4,x,y)


def get_mask(x_array, time_index):
    return x_array.slc.sel(time=x_array.time.values[time_index]).values

def combiner(x_array, compos_list):
    """
    x_array - our dataset
    compos_list - list of time_index photos where first element is the time_index of base array
    """
    base_array = create_composite(x_array,compos_list[0]).astype('int16')
    base_mask = get_mask(x_array,compos_list[0]).astype('int16') 

    #making array mask for base photo
    base_nan = base_mask == 0 
    base_cirrus = base_mask == 10  #cirrus separately to shift it more than other defects
    base_defects = (base_mask == 2) | (base_mask == 3) | (base_mask == 8) | (base_mask == 9)

    
    base_defects = dilation(base_defects, square(23))#~rank_filter(~base_defects,rank=3, size = 24) # rank=18, size = 6)
    base_cirrus = dilation(base_cirrus, square(35))#~rank_filter(~base_cirrus, rank=3, size = 36) #rank=24, size = 3)

    base_defects = base_defects | base_cirrus
    base_cirrus = None
    
    for itr in compos_list[1:]: # ТУТ ХЗ, коли базова фотка буде друга,  то цикл не буде брать першу 
        # можливе рішення [x for x in range(10) if x != 5]
        
        #open new photo
        new_array = create_composite(x_array,time_index = itr ).astype('int16')
        new_mask = get_mask(x_array,time_index = itr).astype('int16')


        #making array mask for new photo
        new_nan = new_mask ==0
        new_cirrus =  (new_mask == 10)
        new_defects = (new_mask == 2)  | (new_mask == 3)  | (new_mask == 8)  | (new_mask == 9)  

        
        new_defects = dilation(new_defects, square(23))#~rank_filter(~new_defects,rank=3, size = 24)# rank=3, size = 6) 
        new_cirrus =  dilation(new_cirrus,square(35)) #~rank_filter(~new_cirrus, rank=3, size = 36)#rank=2, size = 4)


        new_defects = new_defects | new_cirrus
        new_cirrus = None

        cond = base_defects & ~new_defects
        cond = cond & ~new_nan
        base_defects[cond] = False  #modified base mask
        base_array = np.where(cond, new_array, base_array) #modified base array
        base_array = np.where(base_nan, np.nan, base_array)


        if not base_defects.any(): 
            #print('All defects are exclude')
            #print('Done!')
            break #no more defects
    #else:
        #print('Done!')
        #print('Not enough photos to exclude all the defects.')

    return base_array


"""
def driver(x_array):
    a = time.time()
    d = list(range(len(x_array.time.values)))  # list of index times 
    c = []
    for i in d:
        if i+1 < d[-1]:
            c.append([i,i+1,i+2])
    # now list 'c' looks like [[1,2,3],[2,3,4],...]
    # if you want to take by two photos, just del i+2,then  list 'c' will be looking like [[1,2],[2,3],
    composites = np.empty((len(c),4,x_array.y.shape[0], x_array.x.shape[0]))
    for i in c:
        composites[i[0],:,:,:] = combiner(x_array, compos_list = i) # creating every composite
        print('base', i[0], 'under', i[1:], 'Done!')
    print('TIME FOR ALL COMPOSITES: ',time.time() - a,' seconds')
    return composites
"""
def driver(x_array):
    #print('C2-36 D3-22')
    x_array.slc.values = np.nan_to_num(x_array.slc.values,nan =8)
    
    #x_array.red.values = np.nan_to_num(x_array.red.values,nan = 9999)
    #x_array.green.values = np.nan_to_num(x_array.green.values,nan =9999)
    #x_array.blue.values = np.nan_to_num(x_array.blue.values,nan = 9999)
    
    a = time.time()
    c = []
    for i in range(len(x_array.time.values)):
        for j in range(len(x_array.time.values)):
            if i ==j:
                pass
            else:
                c.append([i,j])
    # now list 'c' looks like [[1,2,3],[2,3,4],...]
    # if you want to take by two photos, just del i+2,then  list 'c' will be looking like [[1,2],[2,3],
    composites = np.empty((len(c),4,x_array.y.shape[0], x_array.x.shape[0]))
    for i in range(len(c)):
        composites[i,:,:,:] = combiner(x_array, compos_list = c[i]) # creating every composite
        #print(c[i], 'Done!')
    print('TIME FOR ALL COMPOSITES: ',time.time() - a,' seconds')
    return composites

def make_pairs (n:int) -> list:
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((i,j))
    return pairs

def make_triplets (n:int) -> list:
    triplets = []
    for i in range(n):
        for j in range(n):
            if i!= j: 
                for k in range(n):
                    if j != k and i != k:
                        triplets.append((i,j,k))
    return triplets

def make_sliding_triplets (n:int) -> list:
    index = list(range(n))
    triplets = []
    
    for i in index[:-2]:
        triplets.append((i, i+1, i+2))
    
    return triplets

def make_sliding_triplets_back (n:int) -> list:
    index = list(range(n))
    triplets = []
    
    for i in index[:-2]:
        triplets.append((i, i+1, i+2))
        triplets.append((i+2, i+1, i))
    
    return triplets

def make_combines(dataset: xr.Dataset, sampler:Callable[[int], list] = make_pairs) -> np.ndarray:
    
    indexes = sampler(len(dataset.time))
    num_combines = len(indexes)
    
    composites = np.empty((num_combines,4, dataset.y.shape[0], dataset.x.shape[0]))    
    for i in range(num_combines):
        composites[i] = combiner(dataset, indexes[i])
              
    return composites
    