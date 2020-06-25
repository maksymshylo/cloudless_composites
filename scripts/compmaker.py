import numpy as np
import warnings
from sys import stdout
from math import ceil
from datacube.storage.masking import mask_invalid_data
from tqdm import tqdm

from scripts.data import drop_threshold_cloudiness
from scripts.combiner import make_combines 


class CompMaker():
    '''
    Class for creating composites with the same shape.
    Works with lazy and eager loaded data. To use laze mode
    chunking specify the chunk size and iterate over the instance.

    Parameters
    ----------
    data : xr.Dataset
        Data loaded in lazy or eager mode.
    start_date : Union[str, datetime]
        Composite start date.
    end_date : Union[str, datetime]
        Composite start date.
    window_size : int
        Number of days between files in output composite.
    chunk_size : Optional[Typle[int, int]]
        Only for lazy loaded data! Size of chunk in output data.
    threshold_cloudiness : float
     Threshold to drop files with bigger cloudiness.
    '''

    def __init__(self, data, start_date, end_date, window_size=10,
                 chunk_size=None, threshold_cloudiness=1):

        #start_date = np.datetime64(start_date)
        #end_date = np.datetime64(end_date) + np.timedelta64(1,'D')
        #time_idx = data.time.data
        #idx = (time_idx >= start_date) & (time_idx < end_date)

        self.data = data
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.threshold_cloudiness = threshold_cloudiness
        self.splited_dates = self._split_dates()
        self.lazy_mode = False

        if chunk_size:  # no chunk given = lazy mode
            self.lazy_mode = True
            self.indexes = self._split_indexes()

    def __len__(self):
        '''
        Only for lazy mode!

        Raises
        ------
        TypeError
        If not in lazy moad.

        Returns
        -------
        Number of chunks in input data.
        '''
        if not self.lazy_mode:
            raise TypeError('__len__ is defined only for lazy loaded data mode.')
        else:
            return len(self.indexes)

    def __getitem__(self, idx: int):
        '''
        Only for lazy mode! Get a composite chunk of the specified size.

        Raises
        ------
        TypeError
        If not in lazy moad.

        Returns
        -------
        The composite chunk of the specified size
        '''
        if not self.lazy_mode:
            raise TypeError('Indexing is defined only for lazy loaded data mode.')

        x_range, y_range = self.indexes[idx]
        chunk = self.data[dict(x=slice(*x_range), y=slice(*y_range))]
        res = self.make_single_comp(chunk)
        
        #print(res.shape)
        #print(self.chunk_size)
        #print(res.shape[-2:] != self.chunk_size)
        
        if res.shape[-2:] != self.chunk_size:
            zer = np.zeros((res.shape[0], *self.chunk_size))
            zer[:, :res.shape[1], :res.shape[2]] = res
            res = zer

        return res, (x_range, y_range)

    def make_single_comp(self, data=None):
        '''
        Make composites over the whole input data.

        Parameters
        ----------
        data: xr.Dataset :
        Don't specify this parameter if you call the function directly.

        Returns
        -------
        res : np.array
        Composites over the whole data. Shape : (n_composites, 4, y, x).
        '''

        if not data:
            data = self.data

        data = data.compute()
        # replace 99999 with 0
        data = mask_invalid_data(data).fillna(0)
        data = drop_threshold_cloudiness(data, self.threshold_cloudiness, save_dims=True)

        dates = self.splited_dates
        res = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            
            for window in dates:
                tmp = make_combines(data.sel(time=window))
                tmp = np.nan_to_num(np.nanmedian(tmp, axis=0))
                res.append(tmp)

        return np.vstack(res)  # shape : (n_composites, 4, y, x)
    '''
    def _split_dates(self):

        idxs = self.data.time.values
        n_iters = ceil((idxs[-1] - idxs[0]) / np.timedelta64(1, 'D')) // self.window_size + 1
        window_start = idxs[0]
        window_end = window_start + np.timedelta64(self.window_size*(1), 'D')
        dates = []  # список массивов дат, разбитых окном

        for i in range(1, n_iters):
            in_window = idxs[(idxs >= window_start) & (idxs < window_end)]
            if in_window.size > 0:
                dates.append(in_window)

            window_start = window_end
            window_end += np.timedelta64(self.window_size, 'D')
            
            return dates
    '''        
    def _split_dates(self):
        window = np.timedelta64(self.window_size,'D')
        dates_to_split = np.array(list(map(lambda x: np.datetime64(str(x)[:10]),
                                           self.data.time.values)))
        
        window_borders = []
        pos = self.start_date
        while pos < self.end_date:
            arr = [pos,pos + window]
            pos = pos + window  
            window_borders.append(arr)
    
        window_borders[-1][-1] = self.end_date
    
        splited_dates = []
        for (window_start, window_end) in window_borders:
            dates_in_window = (dates_to_split >= window_start) & (dates_to_split <= window_end) 
            splited_dates.append(self.data.time.values[dates_in_window])
    
        return splited_dates

        

    def _split_indexes(self):
        x_pix = self.data.x.shape[0]
        y_pix = self.data.y.shape[0]

        x_idx = list(range(0, x_pix, self.chunk_size[0]))
        y_idx = list(range(0, y_pix, self.chunk_size[1]))

        x_idx.append(x_pix)
        y_idx.append(y_pix)

        c = []
        for j in range(len(y_idx)-1):
            for i in range(len(x_idx)-1):
                chunk_x = (x_idx[i], x_idx[i+1])
                chunk_y = (y_idx[j], y_idx[j+1])
                c.append([chunk_x,chunk_y])

        self.matrix_of_chunks = np.array(list(range(len(c)))).reshape((len(y_idx)-1, len(x_idx)-1))
        print("Matrix of chunks: ")
        print(self.matrix_of_chunks)

        return c
