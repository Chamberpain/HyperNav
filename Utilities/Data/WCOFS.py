from HyperNav.Utilities.Data.UVBase import Base, UVTimeList
from GeneralUtilities.Plot.Cartopy.regional_plot import CCSCartopy
from GeneralUtilities.Compute.Depth.depth_utilities import ETopo1Depth
import numpy as np 
import datetime
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
file_handler = FilePathHandler(ROOT_DIR,'WGOFS')
from pydap.client import open_url
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList, flat_list
from urllib.error import HTTPError
from socket import timeout
import os
import pickle
import shapely.geometry
import gsw
import matplotlib.pyplot as plt
from urllib.error import HTTPError
import os, shutil
from netCDF4 import Dataset
from GeneralUtilities.Data.pickle_utilities import load
from HyperNav.Utilities.Data.XROM_Utilities import return_dims,dataset_time


class WCOFSBase(Base):
	dataset_description = 'WGOFS'
	hours_list = np.arange(0,73,3).tolist()
	time_step = datetime.timedelta(hours=3)
	file_handler = file_handler
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	@classmethod
	def get_dataset(cls,datetime_instance):
		today = datetime.date.today()
		delta = datetime_instance-datetime.datetime(today.year,today.month,today.day)
		hours_delta = int(delta.days*24+delta.seconds/3600)
		try:
			path = today.strftime('https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS/%Y/%m/%d/wcofs.t03z.%Y%m%d.regulargrid.f{0:03d}.nc'.format(hours_delta-3))
			dataset = open_url(path)
		except:
			path = today.strftime('https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS/%Y/%m/%d/wcofs.t03z.%Y%m%d.regulargrid.n{0:03d}.nc'.format(hours_delta+21))
			dataset = open_url(path)			
		time_since = datetime.datetime.strptime(dataset['time'].attributes['units'],'seconds since %Y-%m-%d %H:%M:%S')
		time = UVTimeList.time_list_from_seconds(dataset['time'][:],time_since)[0]
		assert datetime_instance==time
		return dataset

	@classmethod
	def get_dimensions(cls,urlon,lllon,urlat,lllat,max_depth,dataset):
		time_since = datetime.datetime.strptime(dataset['time'].attributes['units'],'seconds since %Y-%m-%d %H:%M:%S')
		time_start = UVTimeList.time_list_from_seconds(dataset['time'][:],time_since)[0]
		time = [time_start + cls.time_step*x for x in range(len(cls.hours_list))]
		lats = LatList(flat_list(dataset['Latitude'][:,0].data))
		lons = LonList(flat_list(dataset['Longitude'][0,:].data))
		depth = DepthList(-dataset['Depth'][:].data)
		depth_idx = depth.find_nearest(max_depth,idx=True)
		depth = depth[:(depth_idx+1)]
		higher_lon_idx = lons.find_nearest(urlon,idx=True)
		lower_lon_idx = lons.find_nearest(lllon,idx=True)
		lons = lons[lower_lon_idx:higher_lon_idx]
		higher_lat_idx = lats.find_nearest(urlat,idx=True)
		lower_lat_idx = lats.find_nearest(lllat,idx=True)
		lats = lats[lower_lat_idx:higher_lat_idx]
		units = dataset['u_eastward'].attributes['units']
		return (time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,units,time_since)

	@classmethod
	def download_and_save(cls):
		folder = os.path.dirname(cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'))
		for filename in os.listdir(folder):
		    file_path = os.path.join(folder, filename)
		    try:
		        if os.path.isfile(file_path) or os.path.islink(file_path):
		            os.unlink(file_path)
		        elif os.path.isdir(file_path):
		            shutil.rmtree(file_path)
		    except Exception as e:
		        print('Failed to delete %s. Reason: %s' % (file_path, e))

		for time in cls.dataset_time:
			dataset = cls.get_dataset(time)
			k_filename = cls.file_handler.tmp_file(cls.make_k_filename(time))
			folder = os.path.dirname(k_filename)
			if not os.path.exists(folder):
			    os.makedirs(folder)

			u_holder = dataset['u_eastward'][0
			,:(len(cls.depths))
			,cls.lllat_idx:cls.urlat_idx
			,cls.lllon_idx:cls.urlon_idx]
			v_holder = dataset['v_northward'][0
			,:(len(cls.depths))
			,cls.lllat_idx:cls.urlat_idx
			,cls.lllon_idx:cls.urlon_idx]
			with open(k_filename, 'wb') as f:
				pickle.dump({'u':u_holder.data[0],'v':v_holder.data[0], 'time':time},f)
			f.close()

	@classmethod
	def load(cls):
		u_list = []
		v_list = []
		time_list = []
		for time in cls.dataset_time:
			k_filename = cls.make_k_filename(time)
			try:
				with open(k_filename, 'rb') as f:
					uv_dict = pickle.load(f)
			except FileNotFoundError:
				print(k_filename+' not found')
				continue
			u_list.append(uv_dict['u'])
			v_list.append(uv_dict['v'])
			time_list.append(uv_dict['time'])
		u = np.array(u_list)
		v = np.array(v_list)

		assert u.shape==v.shape
		assert u.shape[0] == len(time_list)
		assert u.shape[1] == len(cls.depths)
		assert u.shape[2] == len(cls.lats)
		assert u.shape[3] == len(cls.lons)
		out = cls(u=u*cls.scale_factor,v=v*cls.scale_factor,time=time_list)
		return out


class WCOFSSouthernCalifornia(WCOFSBase):
	location='SoCal'
	facecolor = 'Pink'
	urlat = 35
	lllat = 30
	lllon = -122
	urlon = -116.5
	max_depth = -700
	PlotClass = CCSCartopy
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	DepthClass = ETopo1Depth
	dataset = WCOFSBase.get_dataset(datetime.datetime(datetime.date.today().year,datetime.date.today().month,datetime.date.today().day,0))
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = WCOFSBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

	@classmethod
	def get_dataset_shape(cls):
		longitude = cls.dataset['longitude'][:].data
		longitude[longitude>180]=longitude[longitude>180]-360
		lllat = min(cls.dataset['latitude'][:])
		urlat = max(cls.dataset['latitude'][:])
		lllon = min(longitude)
		urlon = max(longitude)
		ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
		return ocean_shape

class WCOFSSouthernCaliforniaHistorical(WCOFSSouthernCalifornia):
	#this needs to be hard coded for the historical runs
	location='SoCal'
	facecolor = 'Pink'
	urlat = 33.7
	lllat = 32.5
	lllon = -118
	urlon = -117
	max_depth = -500
	PlotClass = CCSCartopy
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	DepthClass = ETopo1Depth
	dataset = WCOFSBase.get_dataset(datetime.datetime(datetime.date.today().year,datetime.date.today().month,datetime.date.today().day,0))
	lats,lons,depths = return_dims(lllon,urlon,lllat,urlat,max_depth)
	lats = LatList(lats.tolist())
	lons = LonList(lons.tolist())
	depths = DepthList(depths.tolist())
	ref_date = datetime.datetime.strptime(dataset['time'].attributes['units'],'seconds since %Y-%m-%d %H:%M:%S')
	urlon_idx = lons.find_nearest(urlon,idx=True)
	lllon_idx = lons.find_nearest(lllon,idx=True)
	urlat_idx = lats.find_nearest(urlat,idx=True)
	lllat_idx = lats.find_nearest(lllat,idx=True)
	units = dataset['u_eastward'].attributes['units']
	dataset_time = TimeList(dataset_time)

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	@classmethod
	def load(cls):
		folder_name = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_historical_data')
		u = np.load(os.path.join(folder_name,'u.npy'))
		v = np.load(os.path.join(folder_name,'v.npy'))
		out = cls(u=u*cls.scale_factor,v=v*cls.scale_factor,time=TimeList(dataset_time))
		return out



class WCOFSMonterey(WCOFSBase):
	location='Monterey'
	facecolor = 'Brown'
	urlat = 39
	lllat = 34
	lllon = -126
	urlon = -121.5
	max_depth = -700
	PlotClass = CCSCartopy
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'HYCOM_reg7_latest3d'
	DepthClass = ETopo1Depth
	dataset = WCOFSBase.get_dataset(datetime.datetime(datetime.date.today().year,datetime.date.today().month,datetime.date.today().day,0))
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = WCOFSBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

	@classmethod
	def get_dataset_shape(cls):
		longitude = cls.dataset['longitude'][:].data
		longitude[longitude>180]=longitude[longitude>180]-360
		lllat = min(cls.dataset['latitude'][:])
		urlat = max(cls.dataset['latitude'][:])
		lllon = min(longitude)
		urlon = max(longitude)
		ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
		return ocean_shape


class WCOFSMontereyHistorical(WCOFSMonterey):
	urlat = 39
	lllat = 34
	lllon = -126
	urlon = -121.5
	max_depth = -700
	#this needs to be hard coded for the historical runs
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	dataset = WCOFSBase.get_dataset(datetime.datetime(datetime.date.today().year,datetime.date.today().month,datetime.date.today().day,0))
	lats,lons,depths = return_dims(lllon,urlon,lllat,urlat,max_depth)
	lats = LatList(lats.tolist())
	lons = LonList(lons.tolist())
	depths = DepthList(depths.tolist())
	ref_date = datetime.datetime.strptime(dataset['time'].attributes['units'],'seconds since %Y-%m-%d %H:%M:%S')
	urlon_idx = lons.find_nearest(urlon,idx=True)
	lllon_idx = lons.find_nearest(lllon,idx=True)
	urlat_idx = lats.find_nearest(urlat,idx=True)
	lllat_idx = lats.find_nearest(lllat,idx=True)
	units = dataset['u_eastward'].attributes['units']
	dataset_time = TimeList(dataset_time)

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	@classmethod
	def load(cls):
		folder_name = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_historical_data')
		u = np.load(os.path.join(folder_name,'u.npy'))
		v = np.load(os.path.join(folder_name,'v.npy'))
		out = cls(u=u*cls.scale_factor,v=v*cls.scale_factor,time=TimeList(dataset_time))
		return out
