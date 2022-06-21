from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import numpy as np
from HyperNav.Data.__init__ import ROOT_DIR
import datetime


class HypernavFileHandler(FilePathHandler):
    """"" Extension of the basic file path handler class with specific filepaths relevent to the Hypernav project
    """""
    @staticmethod
    def nc_file(filename = 'Uniform_out'):
        return ROOT_DIR + '/../Pipeline/Compute/RunParcels/tmp/'+filename+'.nc'


class FloatBehavior():
    vertical_speed = 0.076
    def __init__(self,drift_depth,lat,lon,date_start,date_end):
        self.drift_depth = drift_depth
        self.lat = lat
        self.lon = lon
        self.start_time = date_start.timestamp()
        self.end_time = date_end.timestamp()


    def make_cfg(self,surface_time,total_cycle_time,target_lat=np.nan,target_lon=np.nan):
        argo_cfg = {'lat': self.lat, 'lon': self.lon, 'target_lat': target_lat, 'target_lon': target_lon,
                    'time': self.start_time, 'end_time': self.end_time, 'depth': 10, 'min_depth': 10, 'drift_depth': abs(self.drift_depth),
                    'max_depth': abs(self.max_depth),
                    'surface_time': surface_time, 'total_cycle_time': total_cycle_time,
                    'vertical_speed': self.vertical_speed,
                    }
        return argo_cfg

class HyperNavBehavior(FloatBehavior):
    total_cycle_time = 24*60*60
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.max_depth = self.drift_depth

    def no_hypernav(self):
        surface_time = 60*15
        return ('no hypernav',self.make_cfg(surface_time,self.total_cycle_time))

    def normal(self):
        surface_time = 60*90
        return ('normal',self.make_cfg(surface_time,self.total_cycle_time))

    def long_trans(self):
        surface_time = 60*300
        return ('long transmission',self.make_cfg(surface_time,self.total_cycle_time))

    def two_day_cycle(self):
        surface_time = 60*90
        return ('two day cycle',self.make_cfg(surface_time,2*self.total_cycle_time))
