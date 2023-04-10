'''
Classes and functions to download and parse HYCOM dataset as a pickle file
for the float trajectory prediction.
'''

# 2023-04-10 rl Initial code for Seatrec project


import os
import glob
import pickle
# import logging
import numpy as np
from pydap.client import open_url
from datetime import datetime, timedelta


class Source:
    MODEL_NAME = 'HYCOM'
    SOURCE_NAME = 'TDS'
    BASE_URL = ''
    LAT, LON = 'lat', 'lon'


    def __init__(self, id: str = ''):
        print(self.BASE_URL + id)
        url = self.BASE_URL + id
        self.dataset = open_url(url)


    def get_times(self):
        return np.array([t for t in self.dataset['time']], dtype=np.int32)


    def get_datetime_origin(self):
        return datetime.strptime(self.dataset.attributes['NC_GLOBAL']['time_origin'], '%Y-%m-%d %H:%M:%S')


    def get_filename(self, path_to_wk, filename_prefix):
        return os.path.join(path_to_wk, f"{filename_prefix}.{self.get_datetime_origin().strftime('%Y%m%d%HZ')}.pickle")


    def download(self, box: dict, path_to_wk: str = None, filename_midefix: str = None, filename_prefix: str = None,
                 forecast_duration: timedelta = timedelta(days=5), max_depth: int = 700,
                 force: bool = False):
        
        if path_to_wk is None:
            path_to_wk = os.getcwd()
        if filename_prefix is None:
            filename_prefix = f'{self.MODEL_NAME}.{self.SOURCE_NAME}'
            if filename_midefix:
                filename_prefix += f'.{filename_midefix}'

        local_filename = self.get_filename(path_to_wk, filename_prefix)
        if not force and os.path.exists(local_filename):
            # logger.debug('Forecast field already downloaded.')
            with open(local_filename, 'rb') as f:
                field = pickle.load(f)
            return field, local_filename

        # Get time window to download
        ts = self.get_times()
        now = datetime.utcnow()
        dt_end = now + forecast_duration
        dt_start_idx = find_nearest(ts, now.timestamp())
        dt_end_idx = find_nearest(ts, dt_end.timestamp())
        ts = ts[dt_start_idx:dt_end_idx]
        # if dt_start_idx == 0 and abs(now - datetime.fromtimestamp(ts[0])) > timedelta(hours=12):
        #     logger.warning('Forecast model data start too late.')
        # if dt_end_idx == len(ts) - 1 and abs(dt_end - datetime.fromtimestamp(ts[-1])) > timedelta(hours=12):
        #     logger.warning('Forecast model data end too early.')

        # Get box to download
        lat = np.array(self.dataset[self.LAT]).astype(np.float32)
        lon = np.array(self.dataset[self.LON]).astype(np.float32)
        lllat_idx, urlat_idx = find_nearest(lat, box['lllat']), find_nearest(lat, box['urlat'])
        lat = lat[lllat_idx:urlat_idx]
        lllon_idx, urlon_idx = find_nearest(lon, box['lllon'] % 360), find_nearest(lon, box['urlon'] % 360)
        lon = lon[lllon_idx:urlon_idx]
        depth = np.array(self.dataset['depth']).astype(np.float32)
        deep_depth_idx = find_nearest(depth, max_depth)
        depth = depth[:deep_depth_idx + 1]

        # Download Eastward Water Velocity (u, m/s), Northward Water Velocity (v, m/s)
        u = (self.dataset['water_u'][dt_start_idx:dt_end_idx, 0:deep_depth_idx + 1, lllat_idx:urlat_idx, lllon_idx:urlon_idx].data)[0]

        # logger.debug('Downloading water_v')
        v = (self.dataset['water_v'][dt_start_idx:dt_end_idx, 0:deep_depth_idx + 1, lllat_idx:urlat_idx, lllon_idx:urlon_idx].data)[0]

        # Write output
        field = {'data': {'U': u, 'V': v, 'W': np.zeros(u.shape)},
                 'dimensions': {'time': ts, 'depth': depth, 'lat': lat, 'lon': lon}}
        with open(local_filename, 'xb') as f:
            pickle.dump(field, f)

        return field, local_filename


class SourceNOAA(Source):
    # New forecast runs are typically available at 00:00Z
    SOURCE_NAME = 'NOAA'
    BASE_URL = 'https://www.ncei.noaa.gov/erddap/griddap/'
    LAT, LON = 'latitude', 'longitude'

    def __init__(self, id: str = 'HYCOM_reg7_latest3d'):
        super().__init__(id)


'''
Helper Methods
''' 
def find_nearest(array, value):
    """
    Find nearest neighbor and return its index
    :param array:
    :param value:
    :return:
    """
    idx = (np.abs(array - value)).argmin()
    return int(idx)


#<unittests>
import unittest

class test_hycom(unittest.TestCase):

    def test_constructs(self) :
        test_folder = './test'
        cfg = dict(forecast_duration=timedelta(days=5), max_depth=700)
        hi_box = {'lllat': 16, 'lllon': -159, 'urlat': 22, 'urlon': -154}
        SourceNOAA('HYCOM_reg6_latest3d').download(hi_box, test_folder, 'HI', **cfg)

        # Cleanup
        if os.path.exists(test_folder):
            fileList = glob.glob(test_folder + '/*.pickle')
            
            for file in fileList:
                try:
                    os.remove(file)
                except OSError:
                    print("Error while deleting file")

if __name__ == "__main__" :
    # Run the unittests
    unittest.main()

#</unittests>
