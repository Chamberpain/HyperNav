from HyperNav.Utilities.Data.UVBase import Base, UVTimeList
from GeneralUtilities.Plot.Cartopy.regional_plot import GOMCartopy,CreteCartopy,KonaCartopy,PuertoRicoCartopy, TahitiCartopy
from GeneralUtilities.Compute.Depth.depth_utilities import PACIOOS,ETopo1Depth
import numpy as np 
import datetime
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
file_handler = FilePathHandler(ROOT_DIR,'HYCOMBase')
from pydap.client import open_url
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList, flat_list
from urllib.error import HTTPError
from socket import timeout
import os
import pickle
import shapely.geometry



class HYCOMBase(Base):
	dataset_description = 'HYCOM'
	hours_list = np.arange(0,25,3).tolist()
	time_step = datetime.timedelta(hours=3)
	base_html = 'https://www.ncei.noaa.gov/erddap/griddap/'
	file_handler = file_handler
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	@classmethod
	def get_dataset(cls,ID):
		return open_url(cls.base_html+ID)

	@classmethod
	def get_dimensions(cls,urlon,lllon,urlat,lllat,max_depth,dataset):
		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%d-%b-%Y %H:%M:%S')
		time = UVTimeList.time_list_from_seconds(dataset['time'][:],time_since)
		time[0] = time[1] - datetime.timedelta(hours = 3)
		lats = LatList(dataset['latitude'][:])
		lons = dataset['longitude'][:].data
		lons[lons>180] = lons[lons>180]-360
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
		idx_list = cls.dataset_time.return_time_list()
		k = 0
		while k < len(idx_list)-1:
			print(k)
			k_filename = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'+str(k))
			folder = os.path.dirname(k_filename)
			if not os.path.exists(folder):
			    os.makedirs(folder)

			if os.path.isfile(k_filename):
				k +=1
				continue
			try:
				u_holder = cls.dataset['water_u'][idx_list[k]:idx_list[k+1]
				,:(len(cls.depths))
				,cls.lllat_idx:cls.urlat_idx
				,cls.lllon_idx:cls.urlon_idx]
				v_holder = cls.dataset['water_v'][idx_list[k]:idx_list[k+1]
				,:(len(cls.depths))
				,cls.lllat_idx:cls.urlat_idx
				,cls.lllon_idx:cls.urlon_idx]
				with open(k_filename, 'wb') as f:
					pickle.dump({'u':u_holder['water_u'].data,'v':v_holder['water_v'].data, 'time':u_holder['time'].data},f)
				k +=1
			except:
				print('Index ',k,' encountered an error and did not save. Trying again')
				cls.dataset = HYCOMBase.get_dataset(cls.ID)
				continue

	@classmethod
	def download_recent(cls):
		idx_list = cls.dataset_time.return_time_list()
		k = len(idx_list)-10
		while k < len(idx_list)-1:
			print(k)
			k_filename = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'+str(k))
			folder = os.path.dirname(k_filename)
			if not os.path.exists(folder):
			    os.makedirs(folder)

			if os.path.isfile(k_filename):
				k +=1
				continue
			try:
				u_holder = cls.dataset['water_u'][idx_list[k]:idx_list[k+1]
				,:(len(cls.depths))
				,cls.lllat_idx:cls.urlat_idx
				,cls.lllon_idx:cls.urlon_idx]
				v_holder = cls.dataset['water_v'][idx_list[k]:idx_list[k+1]
				,:(len(cls.depths))
				,cls.lllat_idx:cls.urlat_idx
				,cls.lllon_idx:cls.urlon_idx]
				with open(k_filename, 'wb') as f:
					pickle.dump({'u':u_holder['water_u'].data,'v':v_holder['water_v'].data, 'time':u_holder['time'].data},f)
				k +=1
			except:
				print('Index ',k,' encountered an error and did not save. Trying again')
				cls.dataset = HYCOMBase.get_dataset(cls.ID)
				continue

class HYCOMAlaska(HYCOMBase):
	location='Alaska'
	facecolor = 'brown'
	urlat = 70
	lllat = 60
	lllon = -150
	urlon = -140
	max_depth = -700
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'HYCOM_reg17_latest3d'
	DepthClass = ETopo1Depth
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

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

class HYCOMSouthernCalifornia(HYCOMBase):
	location='SoCal'
	facecolor = 'Pink'
	urlat = 35
	lllat = 30
	lllon = -122
	urlon = -116
	max_depth = -700
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'HYCOM_reg7_latest3d'
	DepthClass = ETopo1Depth
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)
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

class HYCOMGOM(HYCOMBase):
	location='GOM'
	urlat = 27.5
	lllat = 25.5
	lllon = -93.5
	urlon = -90.5
	max_depth = -2500
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'HYCOM_reg1_latest3d'
	PlotClass = GOMCartopy
	DepthClass = ETopo1Depth
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class HYCOMMonterey(HYCOMBase):
	location='Monterey'
	facecolor = 'Pink'
	urlat = 39
	lllat = 34
	lllon = -126
	urlon = -121.5
	max_depth = -700
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'HYCOM_reg7_latest3d'
	DepthClass = ETopo1Depth
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

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


class HYCOMGOM(HYCOMBase):
	location='GOM'
	urlat = 28
	lllat = 26
	lllon = -93
	urlon = -90.5
	max_depth = -2500
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'HYCOM_reg1_latest3d'
	PlotClass = GOMCartopy
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
	ID = 'HYCOM_reg6_latest3d'
	PlotClass = KonaCartopy
	DepthClass = PACIOOS
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

	@classmethod
	def get_dataset_shape(cls):
		longitude = cls.dataset['longitude'][:].data
		lllat = min(cls.dataset['latitude'][:])
		urlat = max(cls.dataset['latitude'][:])
		lllon = min(longitude)
		urlon = max(longitude)
		ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
		return ocean_shape

class HYCOMPuertoRico(HYCOMBase):
	location = 'PuertoRico'
	facecolor = 'yellow'
	urlon = -65
	lllon = -68.5 
	urlat = 22.5
	lllat = 16
	max_depth = -700
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'HYCOM_reg1_latest3d'
	PlotClass = PuertoRicoCartopy
	DepthClass = ETopo1Depth
	dataset = HYCOMBase.get_dataset(ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = HYCOMBase.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

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

# class HYCOMTDS(Base):
# 	dataset_description = 'HYCOM'
# 	hours_list = np.arange(0,25,3).tolist()
# 	file_handler = file_handler

# 	hindcast = [
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1994',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1995',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1996',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1997',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1998',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1999',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2000',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2001',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2002',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2003',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2004',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2005',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2006',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2007',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2008',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2009',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2010',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2011',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2012',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2013',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2014',
# 	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2015',
# 	]
# 	forecast = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'

# 	def __init__(self,*args,**kwargs):
# 		super().__init__(*args,**kwargs)
# 		# self.time.set_ref_date(datetime.datetime(1970,1,1))


# 	@classmethod
# 	def get_dimensions(cls):
# 		dataset = open_url(cls.hindcast[0])
# 		#open dataset using erdap server
# 		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%Y-%m-%d %H:%M:%S')
# 		UVTimeList.set_ref_date(time_since)
# 		time = UVTimeList.time_list_from_seconds(dataset['time'][:])
# 		lats = LatList(dataset['lat'][:])
# 		lons = dataset['lon'][:].data
# 		lons[lons>180] = lons[lons>180]-360
# 		lons = LonList(lons)
# 		depth = -dataset['depth'][:].data
# 		depth = DepthList(depth)
# 		depth_idx = depth.find_nearest(-2000,idx=True)
# 		depth = depth[:(depth_idx+1)]
# 		higher_lon_idx = lons.find_nearest(cls.urlon,idx=True)
# 		lower_lon_idx = lons.find_nearest(cls.lllon,idx=True)
# 		lons = lons[lower_lon_idx:higher_lon_idx]
# 		higher_lat_idx = lats.find_nearest(cls.urlat,idx=True)
# 		lower_lat_idx = lats.find_nearest(cls.lllat,idx=True)
# 		lats = lats[lower_lat_idx:higher_lat_idx]
# 		return (time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,dataset['water_u'].units)


# 	@classmethod
# 	def recompile_hindcast(cls,years_to_compile):
# 		assert isinstance(years_to_compile,list) 
# 		assert isinstance(years_to_compile[0],int) 

# 		url = cls.hindcast[0]
# 		dataset = open_url(url)
# 		lats = dataset['lat'][:].data
# 		lats = LatList(lats.tolist())
# 		lons = dataset['lon'][:].data
# 		lons = LonList(lons.tolist())
# 		depths = dataset['depth'][:].data
# 		depth_idx = depths.tolist().index(2000)
# 		depths=-depths[:depth_idx]
# 		time = dataset['time'][:].data.tolist()

# 		lllon_idx = lons.find_nearest(cls.lllon,idx=True)
# 		urlon_idx = lons.find_nearest(cls.urlon,idx=True)
# 		lllat_idx = lats.find_nearest(cls.lllat,idx=True)
# 		urlat_idx = lats.find_nearest(cls.urlat,idx=True)

# 		lons = lons[lllon_idx:urlon_idx]
# 		lats = lats[lllat_idx:urlat_idx]

# 		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%Y-%m-%d %H:%M:%S')

# 		base_folder = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/')
# 		u_list = []
# 		v_list = []
# 		time_list = []

# 		for file_ in sorted(os.listdir(base_folder)):
# 			filename = os.path.join(base_folder,file_)
# 			if filename.split('/')[-1]=='.DS_Store':
# 				continue
# 			year = int(filename.split('/')[-1].split('_')[0])
# 			if year in years_to_compile:
# 				with open(filename, 'rb') as f:
# 					print(filename)
# 					holder = pickle.load(f)
# 				time = TimeList([time_since+datetime.timedelta(hours = x) for x in holder['time']])
# 				mask = [x.year==year for x in time]
# 				time_holder = [x for x in time if x.year==year]
# 				u_holder = holder['v'][mask]
# 				v_holder = holder['u'][mask]

# 				assert u_holder.shape == v_holder.shape
# 				assert u_holder.shape[0] == len(time_holder)

# 				if time_holder:
# 					time_list.append(time_holder)
# 					u_list.append(u_holder)
# 					v_list.append(v_holder)


# 		u = np.concatenate([x for _,x in sorted(zip([q[0] for q in time_list],u_list))])
# 		u = u*dataset['water_u'].attributes['scale_factor']
# 		v = np.concatenate([x for _,x in sorted(zip([q[0] for q in time_list],v_list))])
# 		v = v*dataset['water_u'].attributes['scale_factor']
# 		time = sorted(flat_list(time_list))
# 		return cls(u=u,v=v,lons=lons,lats=lats,depth=depths,time=time,units=dataset['water_u'].attributes['units'])


# 	@classmethod
# 	def download_and_save(cls):
# 		for url in cls.hindcast:
# 			year = url.split('/')[-1]
# 			time_idx = 0
# 			dataset = open_url(url)
# 			lats = dataset['lat'][:].data
# 			lats = LatList(lats.tolist())
# 			lons = dataset['lon'][:].data
# 			lons = LonList(lons.tolist())
# 			depths = dataset['depth'][:].data
# 			depth_idx = depths.tolist().index(2000)
# 			time = dataset['time'][:].data.tolist()

# 			lllon_idx = lons.find_nearest(cls.lllon,idx=True)
# 			urlon_idx = lons.find_nearest(cls.urlon,idx=True)
# 			lllat_idx = lats.find_nearest(cls.lllat,idx=True)
# 			urlat_idx = lats.find_nearest(cls.urlat,idx=True)

# 			while time_idx<len(time):
# 				print('time index is ',time_idx)
# 				print('max time index is ',len(time))
# 				k_filename = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'+year+'_'+str(time_idx))
# 				if os.path.isfile(k_filename):
# 					time_idx+=10
# 					continue

# 				try:
# 					u_holder = dataset['water_u'].data[0][time_idx:(time_idx+10),:depth_idx,lllat_idx:urlat_idx,lllon_idx:urlon_idx]
# 					v_holder = dataset['water_v'].data[0][time_idx:(time_idx+10),:depth_idx,lllat_idx:urlat_idx,lllon_idx:urlon_idx]
# 					time_holder = time[time_idx:(time_idx+10)]
# 					with open(k_filename, 'wb') as f:
# 						pickle.dump({'u':u_holder,'v':v_holder, 'time':time_holder},f)
# 					time_idx += 10
# 				except:
# 					continue


# 		#open dataset using erdap server
# 		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%Y-%m-%d %H:%M:%S')
# 		TimeList.set_ref_date(time_since)
# 		time = TimeList.time_list_from_seconds(dataset['time'][:])
# 		lats = LatList(dataset['latitude'][:])
# 		lons = dataset['longitude'][:].data
# 		lons[lons>180] = lons[lons>180]-360
# 		lons = LonList(lons)

# 		depth = -dataset['depth'][:].data
# 		depth = DepthList(depth)
# 		depth_idx = depth.find_nearest(-700,idx=True)
# 		depth = depth[:(depth_idx+1)]

# 		higher_lon_idx = lons.find_nearest(cls.urlon,idx=True)
# 		lower_lon_idx = lons.find_nearest(cls.lllon,idx=True)
# 		lons = lons[lower_lon_idx:higher_lon_idx]

# 		higher_lat_idx = lats.find_nearest(cls.urlat,idx=True)
# 		lower_lat_idx = lats.find_nearest(cls.lllat,idx=True)
# 		lats = lats[lower_lat_idx:higher_lat_idx]

# 		time_idx_list = list(range(len(time))[::100])+[len(time)]
# 		u_list = []
# 		v_list = []
# 		k = 0
# 		while k < len(time_idx_list)-1:
# 			print(k)
# 			try:
# 				u_holder = dataset['water_u'][time_idx_list[k]:time_idx_list[k+1]
# 				,:(depth_idx+1)
# 				,lower_lat_idx:higher_lat_idx
# 				,lower_lon_idx:higher_lon_idx]
# 				v_holder = dataset['water_v'][time_idx_list[k]:time_idx_list[k+1]
# 				,:(depth_idx+1)
# 				,lower_lat_idx:higher_lat_idx
# 				,lower_lon_idx:higher_lon_idx]
# 				u_list.append(u_holder)
# 				v_list.append(v_holder)
# 				k +=1
# 			except:
# 				continue
# 		u = np.concatenate([np.array(x['water_u']) for x in u_list])
# 		v = np.concatenate([np.array(x['water_v']) for x in v_list])
# 		assert u.shape==v.shape
# 		assert u.shape[0] == len(time)
# 		assert u.shape[1] == len(depth)
# 		assert u.shape[2] == len(lats)
# 		assert u.shape[3] == len(lons)
# 		out = cls(u=u,v=v,lons=lons,lats=lats,time=time,depth=depth)
# 		out.save()


# class HYCOMTahiti(HYCOMTDS):
# 	urlon = -148
# 	lllon = -152
# 	urlat = -16
# 	lllat = -19
# 	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
# 	location = 'Tahiti'
# 	ID = 'hawaii_soest_a2d2_f95d_0258'
# 	PlotClass = TahitiCartopy
# 	DepthClass = ETopo1Depth

# class HYCOMCrete(HYCOMTDS):
# 	urlon = 30
# 	lllon = 20
# 	urlat = 41
# 	lllat = 31
# 	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
# 	location = 'Crete'
# 	ID = 'hawaii_soest_a2d2_f95d_0258'
# 	PlotClass = CreteCartopy
# 	DepthClass = ETopo1Depth
