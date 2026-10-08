from GeneralUtilities.Data.Download.legacy_model_download import download_legacy_chunks, open_legacy_dataset
from HyperNav.Utilities.Data.UVBase import Base, UVTimeList
from GeneralUtilities.Plot.Cartopy.regional_plot import GOMCartopy,CreteCartopy,KonaCartopy,PuertoRicoCartopy,CCSCartopy,TahitiCartopy
from GeneralUtilities.Compute.Depth.depth_utilities import PACIOOS,ETopo1Depth
import numpy as np
import datetime
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Download.download_paths import LazyDownloadPaths
file_handler = LazyDownloadPaths(ROOT_DIR,'CurrentHYCOMBase')
from pydap.client import open_url
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList, flat_list
from urllib.error import HTTPError
from socket import timeout
import os
import pickle
import shapely.geometry
import gsw
import matplotlib.pyplot as plt

class HYCOMTimeList(UVTimeList):
	def return_time_list(self):
		return list(range(len(self))[::5])+[len(self)-2] #-1 because of pythons crazy list indexing


class HYCOMBase(Base):
	time_method = HYCOMTimeList.time_list_from_hours
	dataset_description = 'GOFS'
	hours_list = np.arange(0,25,3).tolist()
	time_step = datetime.timedelta(hours=3)
	base_html = 'https://tds.hycom.org/thredds/dodsC/'
	file_handler = file_handler
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	@classmethod
	def get_dataset(cls,ID):
		return open_legacy_dataset(cls.base_html, ID, opener=open_url)

	@classmethod
	def get_dimensions(cls,urlon,lllon,urlat,lllat,max_depth,dataset):
		time_since = datetime.datetime.strptime(dataset['time'].attributes['units'],'hours since %Y-%m-%d %H:%M:%S.000 UTC')
		time = HYCOMTimeList.time_list_from_hours(dataset['time'][:],time_since)
		time[0] = time[1] - datetime.timedelta(hours = 3)
		lats = LatList(dataset['lat'][:])
		lons = dataset['lon'][:].data
		lons[lons>=180] = lons[lons>=180]-360
		lons = LonList(lons)
		depth = -dataset['depth'][:].data
		depth = DepthList(depth)
		depth_idx = depth.find_nearest(max_depth,idx=True)
		depth = depth[:(depth_idx+1)]
		higher_lon_idx = lons.find_nearest(urlon,idx=True)
		lower_lon_idx = lons.find_nearest(lllon,idx=True)
		lons = lons[lower_lon_idx:higher_lon_idx]
		higher_lat_idx = lats.find_nearest(urlat,idx=True)
		lower_lat_idx = lats.find_nearest(lllat,idx=True)
		lats = lats[lower_lat_idx:higher_lat_idx]
		units = dataset['water_u'].attributes['units']
		return (time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,units,time_since)

	@classmethod
	def download_and_save(cls):
		return download_legacy_chunks(cls, direct_data=True, reopen=lambda: HYCOMBase.get_dataset(cls.ID))

	@classmethod
	def download_recent(cls):
		return download_legacy_chunks(cls, recent=True, reopen=lambda: HYCOMBase.get_dataset(cls.ID))

	@classmethod
	def get_sal_temp_profiles(cls,lat,lon,start_date,end_date):
		lon_idx = cls.lons.find_nearest(lon,idx=True)
		lat_idx = cls.lats.find_nearest(lat,idx=True)
		depth_idx = cls.depths.find_nearest(-700,idx=True)
		time_start_idx = cls.dataset_time.find_nearest(start_date,idx=True)
		time_end_idx = cls.dataset_time.find_nearest(end_date,idx=True)

		fig, ax1 = plt.subplots()
		ax1.set_xlabel('Salinity (psu)', color='tab:red')
		ax1.set_ylabel('Depth (m)')
		sal_data = 	cls.dataset['salinity']['salinity'].data[time_start_idx:time_end_idx,:depth_idx,lat_idx,lon_idx]
		sal_data = sal_data.mean(axis=0).flatten()
		ax1.plot(sal_data, cls.depths[:depth_idx], color='tab:red')
		ax1.tick_params(axis='x', labelcolor='tab:red')

		ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis
		ax2.set_xlabel(r'$\theta_0\ (c)$', color='tab:blue')  # we already handled the x-label with ax1
		temp_data = cls.dataset['water_temp']['water_temp'].data[time_start_idx:time_end_idx,:depth_idx,lat_idx,lon_idx]
		temp_data = temp_data.mean(axis=0).flatten()

		ax2.plot(temp_data, cls.depths[:depth_idx], color='tab:blue')
		ax2.tick_params(axis='x', labelcolor='tab:blue')
		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		gsw.p_from_z(cls.depths,lat)
		density = gsw.density.sigma0(sal_data,temp_data)
		fig1, ax1 = plt.subplots()
		ax1.plot(density,cls.depths[:depth_idx])
		plt.xlabel(r'$\sigma_0\ (kg\ m^{-3})$')
		plt.ylabel('Depth (m)')
		return (fig,fig1)



class HYCOMSouthernCalifornia(HYCOMBase):
	location='SoCal'
	facecolor = 'Pink'
	urlat = 35
	lllat = 30
	lllon = -122
	urlon = -116
	max_depth = -700
	PlotClass = CCSCartopy
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])
	ID = 'FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd'
	DepthClass = ETopo1Depth
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)


class HYCOMHawaii(HYCOMBase):
	location='Hawaii'
	facecolor = 'blue'
	urlat = 22
	lllat = 16
	lllon = -159
	urlon = -154
	max_depth = -2500
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])
	ID = 'FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd'
	PlotClass = KonaCartopy
	DepthClass = PACIOOS
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class HYCOMPuertoRico(HYCOMBase):
	location = 'PuertoRico'
	facecolor = 'yellow'
	urlon = -65
	lllon = -68.5
	urlat = 22.5
	lllat = 16
	max_depth = -700
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])
	ID = 'FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd'
	PlotClass = PuertoRicoCartopy
	DepthClass = ETopo1Depth
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)
