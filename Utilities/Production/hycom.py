import os
import pickle
import logging
from datetime import datetime, timedelta

import numpy as np
from pydap.client import open_url
from webob.exc import HTTPError

from particles import find_nearest

logger = logging.getLogger('forecast-model.hycom')


class Source:
    MODEL_NAME = 'HYCOM'
    SOURCE_NAME = 'TDS'
    BASE_URL = ''
    LAT, LON = 'lat', 'lon'

    def __init__(self, id: str = ''):
        self.dataset = open_url(self.BASE_URL + id, output_grid=False)

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
            logger.debug('Forecast field already downloaded.')
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
        if dt_start_idx == 0 and abs(now - datetime.fromtimestamp(ts[0])) > timedelta(hours=12):
            logger.warning('Forecast model data start too late.')
        if dt_end_idx == len(ts) - 1 and abs(dt_end - datetime.fromtimestamp(ts[-1])) > timedelta(hours=12):
            logger.warning('Forecast model data end too early.')

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
        logger.debug('Downloading water_u')
        u = self.dataset['water_u'][dt_start_idx:dt_end_idx, 0:deep_depth_idx + 1,
                                    lllat_idx:urlat_idx, lllon_idx:urlon_idx].data
        logger.debug('Downloading water_v')
        v = self.dataset['water_v'][dt_start_idx:dt_end_idx, 0:deep_depth_idx + 1,
                                    lllat_idx:urlat_idx, lllon_idx:urlon_idx].data

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

    def __init__(self, id: str = 'HYCOM_reg6_latest3d'):
        super().__init__(id)


class SourceFNMOCBest(Source):
    SOURCE_NAME = 'FNMOC-best'
    BASE_URL = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/'

    def __init__(self, id: str = 'GLBy0.08_930_FMRC_best.ncd'):
        super().__init__(id)

    def get_datetime_origin(self):
        dt_origin = datetime.strptime(self.dataset['time_run'].attributes['units'],
                                      'hours since %Y-%m-%d %H:%M:%S.000 UTC')
        return dt_origin + timedelta(hours=self.dataset['time_run'].__array__()[-1])

    def get_times(self):
        dt_origin = datetime.strptime(self.dataset['time'].attributes['units'],
                                      'hours since %Y-%m-%d %H:%M:%S.000 UTC')
        dt = [dt_origin + timedelta(hours=h) for h in self.dataset['time']]
        return np.array([datetime.timestamp(x) for x in dt], dtype=np.int32)


class SourceFNMOCRun(SourceFNMOCBest):
    SOURCE_NAME = 'FNMOC-run'
    BASE_URL = SourceFNMOCBest.BASE_URL + 'runs/'

    def __init__(self, id: str = ''):
        if id == '':
            self.run_dt = (datetime.utcnow() - timedelta(days=1)).replace(hour=12, minute=0, second=0)
            for i in range(3):
                try:
                    id = f"GLBy0.08_930_FMRC_RUN_{self.run_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    self.dataset = open_url(self.BASE_URL + id, output_grid=False)
                    break
                except HTTPError:
                    logger.warning(f"Unable to find run {id}")
                    self.run_dt -= timedelta(days=1)
            else:
                raise HTTPError('Unable to find latest HYCOM FMRC run.')
        else:
            self.run_dt = datetime.strptime(id[-20:], '%Y-%m-%dT%H:%M:%SZ')
            try:
                self.dataset = open_url(self.BASE_URL + id, output_grid=False)
            except HTTPError:
                raise HTTPError(f'Unable to find HYCOM FMRC run {id} requested.')

    def get_datetime_origin(self):
        return self.run_dt


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    wk = '/Users/nils/Data/HyperNAV/TrajectoryPrediction/'
    cfg = dict(forecast_duration=timedelta(days=5), max_depth=700)

    hi_box = {'lllat': 16, 'lllon': -159, 'urlat': 22, 'urlon': -154}
    SourceNOAA('HYCOM_reg6_latest3d').download(hi_box, wk, 'HI', **cfg)
    SourceFNMOCBest().download(hi_box, wk, 'HI', **cfg)
    SourceFNMOCRun().download(hi_box, wk, 'HI', **cfg)

    pr_box = {'lllat': 16, 'lllon': -68.5, 'urlat': 22.5, 'urlon': -65}
    SourceNOAA('HYCOM_reg1_latest3d').download(pr_box, wk, 'PR', **cfg)
    SourceFNMOCBest().download(pr_box, wk, 'PR', **cfg)
    SourceFNMOCRun().download(pr_box, wk, 'PR', **cfg)

    tt_box = {'lllat': -19, 'lllon': -152, 'urlat': -16, 'urlon': -148}
    SourceNOAA('Hycom_sfc_3d').download(tt_box, wk, 'TT', **cfg)
    SourceFNMOCBest().download(tt_box, wk, 'TT', **cfg)
    SourceFNMOCRun().download(tt_box, wk, 'TT', **cfg)
