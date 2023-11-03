from HyperNav.Utilities.__init__ import ROOT_DIR
import h5py
from GeneralUtilities.Data.Filepath.instance import get_data_folder
from GeneralUtilities.Compute.list import LatList,LonList,TimeList
import os
DATA_DIR = os.path.join(get_data_folder(),'Processed/HDF5/clear_sky')
import numpy as np

class ClearSkyBase():
	def __init__(self):
		data = h5py.File(os.path.join(DATA_DIR,self.file))

		self.aot_mean = data['AOT869 (mean)']
		self.aot_std = data['AOT869 (std)']
		self.percent_clear_mean = data['percent clear days (mean)']
		self.percent_clear_std = data['percent clear days (std)']

		self.lat = LatList(data['lat'][:,0].tolist())
		self.lon = LonList(data['lon'][0,:].tolist())

	def return_data(self,mean_field,std_field,lat,lon,month_idx):
		lat_idx = self.lat.find_nearest(lat,idx=True)
		lon_idx = self.lon.find_nearest(lon,idx=True)
		return np.random.normal(mean_field[(month_idx-1),lat_idx,lon_idx],std_field[(month_idx-1),lat_idx,lon_idx])

	def return_aot(self,lat,lon,month_idx):
		return self.return_data(self.aot_mean,self.aot_std,lat,lon,month_idx)<=0.1

	def return_clear_sky(self,lat,lon,month_idx):
		return self.return_data(self.percent_clear_mean,self.percent_clear_std,lat,lon,month_idx)>np.random.uniform(0,100)

	def match_up(self,lat,lon,month_idx):
		return self.return_aot(lat,lon,month_idx)&self.return_clear_sky(lat,lon,month_idx)

class ClearSkyBermuda(ClearSkyBase):
	file = 'Bermuda.h5'

class ClearSkySoCal(ClearSkyBase):
	file = 'california_bight.h5'

class ClearSkyCrete(ClearSkyBase):
	file = 'crete_island.h5'

class ClearSkyHawaii(ClearSkyBase):
	file = 'Hawaii.h5'

class ClearSkyMonterey(ClearSkyBase):
	file = 'Monterey.h5'

class ClearSkyPuertoRico(ClearSkyBase):
	file = 'PuertoRico.h5'

class ClearSkyTahiti(ClearSkyBase):
	file = 'Tahiti.h5'
