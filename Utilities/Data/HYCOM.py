from HyperNav.Utilities.Data.UVBase import Base
from GeneralUtilities.Plot.Cartopy.regional_plot import KonaCartopy,PuertoRicoCartopy, TahitiCartopy
from GeneralUtilities.Data.depth.depth_utilities import PACIOOS,ETopo1Depth
import numpy as np 
import datetime
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
file_handler = FilePathHandler(ROOT_DIR,'HYCOMBase')
from pydap.client import open_url
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList, flat_list
from urllib.error import HTTPError
from socket import timeout
import os
import pickle

class HYCOMBase(Base):
	dataset_description = 'HYCOM'
	hours_list = np.arange(0,25,3).tolist()
	base_html = 'https://www.ncei.noaa.gov/erddap/griddap/'
	file_handler = file_handler
	def __init__(self,*args,**kwargs):
		super().__init__(units='m/s',*args,**kwargs)
		self.time.set_ref_date(datetime.datetime(1970,1,1))

class HYCOMHawaii(HYCOMBase):
	location='Hawaii'
	urlat = 22
	lllat = 16
	lllon = -159
	urlon = -154
	ID = 'HYCOM_reg6_latest3d'
	PlotClass = KonaCartopy
	DepthClass = PACIOOS

class HYCOMPuertoRico(HYCOMBase):
	location = 'PuertoRico'
	urlon = -65
	lllon = -68.5 
	urlat = 22.5
	lllat = 16
	ID = 'HYCOM_reg1_latest3d'
	PlotClass = PuertoRicoCartopy
	DepthClass = ETopo1Depth

class HYCOMTDS(Base):
	dataset_description = 'HYCOM'
	hours_list = np.arange(0,25,3).tolist()
	file_handler = file_handler

	hindcast = [
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1994',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1995',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1996',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1997',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1998',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/1999',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2000',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2001',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2002',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2003',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2004',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2005',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2006',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2007',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2008',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2009',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2010',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2011',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2012',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2013',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2014',
	'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2015',
	]
	forecast = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		# self.time.set_ref_date(datetime.datetime(1970,1,1))


	@classmethod
	def recompile_hindcast(cls,years_to_compile):
		assert isinstance(years_to_compile,list) 
		assert isinstance(years_to_compile[0],int) 

		url = cls.hindcast[0]
		dataset = open_url(url)
		lats = dataset['lat'][:].data
		lats = LatList(lats.tolist())
		lons = dataset['lon'][:].data
		lons = LonList(lons.tolist())
		depths = dataset['depth'][:].data
		depth_idx = depths.tolist().index(2000)
		depths=-depths[:depth_idx]
		time = dataset['time'][:].data.tolist()

		lllon_idx = lons.find_nearest(cls.lllon,idx=True)
		urlon_idx = lons.find_nearest(cls.urlon,idx=True)
		lllat_idx = lats.find_nearest(cls.lllat,idx=True)
		urlat_idx = lats.find_nearest(cls.urlat,idx=True)

		lons = lons[lllon_idx:urlon_idx]
		lats = lats[lllat_idx:urlat_idx]

		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%Y-%m-%d %H:%M:%S')

		base_folder = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/')
		u_list = []
		v_list = []
		time_list = []

		for file_ in sorted(os.listdir(base_folder)):
			filename = os.path.join(base_folder,file_)
			if filename.split('/')[-1]=='.DS_Store':
				continue
			year = int(filename.split('/')[-1].split('_')[0])
			if year in years_to_compile:
				with open(filename, 'rb') as f:
					print(filename)
					holder = pickle.load(f)
				time = TimeList([time_since+datetime.timedelta(hours = x) for x in holder['time']])
				mask = [x.year==year for x in time]
				time_holder = [x for x in time if x.year==year]
				u_holder = holder['v'][mask]
				v_holder = holder['u'][mask]

				assert u_holder.shape == v_holder.shape
				assert u_holder.shape[0] == len(time_holder)

				if time_holder:
					time_list.append(time_holder)
					u_list.append(u_holder)
					v_list.append(v_holder)


		u = np.concatenate([x for _,x in sorted(zip([q[0] for q in time_list],u_list))])
		u = u*dataset['water_u'].attributes['scale_factor']
		v = np.concatenate([x for _,x in sorted(zip([q[0] for q in time_list],v_list))])
		v = v*dataset['water_u'].attributes['scale_factor']
		time = sorted(flat_list(time_list))
		return cls(u=u,v=v,lons=lons,lats=lats,depth=depths,time=time,units=dataset['water_u'].attributes['units'])


	@classmethod
	def download_and_save(cls):
		for url in cls.hindcast:
			year = url.split('/')[-1]
			time_idx = 0
			dataset = open_url(url)
			lats = dataset['lat'][:].data
			lats = LatList(lats.tolist())
			lons = dataset['lon'][:].data
			lons = LonList(lons.tolist())
			depths = dataset['depth'][:].data
			depth_idx = depths.tolist().index(2000)
			time = dataset['time'][:].data.tolist()

			lllon_idx = lons.find_nearest(cls.lllon,idx=True)
			urlon_idx = lons.find_nearest(cls.urlon,idx=True)
			lllat_idx = lats.find_nearest(cls.lllat,idx=True)
			urlat_idx = lats.find_nearest(cls.urlat,idx=True)

			while time_idx<len(time):
				print('time index is ',time_idx)
				print('max time index is ',len(time))
				k_filename = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'+year+'_'+str(time_idx))
				if os.path.isfile(k_filename):
					time_idx+=10
					continue

				try:
					u_holder = dataset['water_u'].data[0][time_idx:(time_idx+10),:depth_idx,lllat_idx:urlat_idx,lllon_idx:urlon_idx]
					v_holder = dataset['water_v'].data[0][time_idx:(time_idx+10),:depth_idx,lllat_idx:urlat_idx,lllon_idx:urlon_idx]
					time_holder = time[time_idx:(time_idx+10)]
					with open(k_filename, 'wb') as f:
						pickle.dump({'u':u_holder,'v':v_holder, 'time':time_holder},f)
					time_idx += 10
				except:
					continue


		#open dataset using erdap server
		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%Y-%m-%d %H:%M:%S')
		TimeList.set_ref_date(time_since)
		time = TimeList.time_list_from_seconds(dataset['time'][:])
		lats = LatList(dataset['latitude'][:])
		lons = dataset['longitude'][:].data
		lons[lons>180] = lons[lons>180]-360
		lons = LonList(lons)

		depth = -dataset['depth'][:].data
		depth = DepthList(depth)
		depth_idx = depth.find_nearest(-700,idx=True)
		depth = depth[:(depth_idx+1)]

		higher_lon_idx = lons.find_nearest(cls.urlon,idx=True)
		lower_lon_idx = lons.find_nearest(cls.lllon,idx=True)
		lons = lons[lower_lon_idx:higher_lon_idx]

		higher_lat_idx = lats.find_nearest(cls.urlat,idx=True)
		lower_lat_idx = lats.find_nearest(cls.lllat,idx=True)
		lats = lats[lower_lat_idx:higher_lat_idx]
		#define necessary variables from self describing netcdf 
		attribute_dict = dataset.attributes['NC_GLOBAL']
		time_end = datetime.datetime.strptime(attribute_dict['time_coverage_end'],'%Y-%m-%dT%H:%M:%SZ')


		time_idx_list = list(range(len(time))[::100])+[len(time)]
		u_list = []
		v_list = []
		k = 0
		while k < len(time_idx_list)-1:
			print(k)
			try:
				u_holder = dataset['water_u'][time_idx_list[k]:time_idx_list[k+1]
				,:(depth_idx+1)
				,lower_lat_idx:higher_lat_idx
				,lower_lon_idx:higher_lon_idx]
				v_holder = dataset['water_v'][time_idx_list[k]:time_idx_list[k+1]
				,:(depth_idx+1)
				,lower_lat_idx:higher_lat_idx
				,lower_lon_idx:higher_lon_idx]
				u_list.append(u_holder)
				v_list.append(v_holder)
				k +=1
			except:
				continue
		u = np.concatenate([np.array(x['water_u']) for x in u_list])
		v = np.concatenate([np.array(x['water_v']) for x in v_list])
		assert u.shape==v.shape
		assert u.shape[0] == len(time)
		assert u.shape[1] == len(depth)
		assert u.shape[2] == len(lats)
		assert u.shape[3] == len(lons)
		out = cls(u=u,v=v,lons=lons,lats=lats,time=time,depth=depth)
		out.save()


class HYCOMTahiti(HYCOMTDS):
	urlon = -148
	lllon = -152
	urlat = -16
	lllat = -19
	location = 'Tahiti'
	ID = 'hawaii_soest_a2d2_f95d_0258'
	PlotClass = TahitiCartopy
	DepthClass = ETopo1Depth
