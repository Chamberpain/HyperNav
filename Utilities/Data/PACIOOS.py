import datetime
import numpy as np
from GeneralUtilities.Compute.list import LatList, LonList, DepthList, flat_list
from HyperNav.Utilities.Data.UVBase import Base,UVTimeList
from GeneralUtilities.Compute.Depth.depth_utilities import PACIOOS
from GeneralUtilities.Plot.Cartopy.eulerian_plot import HypernavCartopy
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import os
import requests
from pydap.client import open_url
from GeneralUtilities.Plot.Cartopy.regional_plot import KonaCartopy
import shapely.geometry
import pickle

class PACIOOS(Base):
	dataset_description = 'PACIOOS'
	base_html = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap/'
	DepthClass = PACIOOS
	file_handler = FilePathHandler(ROOT_DIR,'PACIOOS')
	hours_list = np.arange(0,25,3).tolist()
	time_step = datetime.timedelta(hours=3)
	facecolor = 'green'

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)



	@classmethod
	def download_and_save(cls):
		idx_list = cls.dataset_time.return_time_list()
		k = 0
		while k < len(idx_list)-1:
			print(k)
			k_filename = cls.make_k_filename(k)
			if os.path.isfile(k_filename):
				k +=1
				continue
			u_holder = cls.dataset['u'][idx_list[k]:idx_list[k+1]
			,:(len(cls.depths))
			,cls.lllat_idx:cls.urlat_idx
			,cls.lllon_idx:cls.urlon_idx]
			v_holder = cls.dataset['v'][idx_list[k]:idx_list[k+1]
			,:(len(cls.depths))
			,cls.lllat_idx:cls.urlat_idx
			,cls.lllon_idx:cls.urlon_idx]
			with open(k_filename, 'wb') as f:
				pickle.dump({'u':u_holder['u'].data,'v':v_holder['v'].data, 'time':u_holder['time'].data},f)
			k +=1
			continue

	@classmethod
	def get_dataset_shape(cls):
		lllat = min(cls.dataset['latitude'][:])
		urlat = max(cls.dataset['latitude'][:])
		lllon = min(cls.dataset['longitude'][:])
		urlon = max(cls.dataset['longitude'][:])
		ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
		return ocean_shape

	@classmethod
	def get_dimensions(cls,urlon,lllon,urlat,lllat,max_depth,dataset):
		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%d-%b-%Y %H:%M:%S')
		time = UVTimeList.time_list_from_seconds(dataset['time'][:].data.tolist(),time_since)
		lats = LatList(dataset['latitude'][:].data.tolist())
		lons = LonList(dataset['longitude'][:].data.tolist())
		depths = DepthList([-x for x in dataset['depth'][:].data.tolist()])
		depth_idx = depths.find_nearest(max_depth,idx=True)
		depths=depths[:depth_idx]
		lllon_idx = lons.find_nearest(lllon,idx=True)
		urlon_idx = lons.find_nearest(urlon,idx=True)
		lllat_idx = lats.find_nearest(lllat,idx=True)
		urlat_idx = lats.find_nearest(urlat,idx=True)
		lons = lons[lllon_idx:urlon_idx]
		lats = lats[lllat_idx:urlat_idx]
		units = dataset['u'].attributes['units']
		return (time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,time_since)

	@classmethod
	def get_dataset(cls,ID):
		return open_url(cls.base_html+ID)

	def ReturnPACIOOSWaves(self):
		ID = 'ww3_hawaii'
		dataset = self.get_dataset(ID)

		end_time_idx = self.time_idx+len(self.hours_list)*days

		waves = self.dataset['Thgt'][self.time_idx:end_time_idx
		,:
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx].data[0]


	def ReturnPACIOOSWeather(self):
		hours_list = np.arange(0,25,1).tolist()
		ID = 'wrf_hi'
		def __init__(self,float_pos_dict,*args,days=5,**kwargs):
			super().__init__(float_pos_dict,*args,**kwargs)
			end_time_idx = self.time_idx+len(self.hours_list)*days
			self.U,time,self.lats,self.lons = self.dataset['Uwind'][self.time_idx:end_time_idx
			,self.lower_lat_idx:self.higher_lat_idx
			,self.lower_lon_idx:self.higher_lon_idx].data
			self.time = TimeList.time_list_from_seconds(time)


			self.V = self.dataset['Vwind'][self.time_idx:end_time_idx
			,self.lower_lat_idx:self.higher_lat_idx
			,self.lower_lon_idx:self.higher_lon_idx].data[0]

			self.rain = self.dataset['rain'][self.time_idx:end_time_idx
			,self.lower_lat_idx:self.higher_lat_idx
			,self.lower_lon_idx:self.higher_lon_idx].data[0]


class KonaPACIOOS(PACIOOS):
	location='Hawaii'
	urlat = 22
	lllat = 18.5
	lllon = -159
	urlon = -154
	max_depth = -2500
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	PlotClass = KonaCartopy
	ID = 'roms_hiig'
	dataset = PACIOOS.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = PACIOOS.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)


