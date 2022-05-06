from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from HyperNav.Utilities.Data.UVBase import Base,UVTimeList
from GeneralUtilities.Compute.list import LatList, LonList, DepthList, flat_list
from urllib.error import HTTPError
from socket import timeout
import matplotlib.pyplot as plt
import gsw
import os
import pickle
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
from GeneralUtilities.Plot.Cartopy.regional_plot import CreteCartopy
from GeneralUtilities.Data.depth.depth_utilities import ETopo1Depth
import datetime
import gsw
import shapely.geometry
import numpy as np

class CopUVTimeList(UVTimeList):
	def return_time_list(self):
		return list(range(len(self))[::3])+[len(self)-1] #-1 because of pythons crazy list indexing



class Copernicus(Base):
	facecolor = 'orange'
	dataset_description = 'Copernicus'
	base_html = 'https://nrt.cmems-du.eu/thredds/dodsC/'
	time_step = datetime.timedelta(hours=1)
	hours_list = np.arange(0,25,1).tolist()
	DepthClass = ETopo1Depth
	file_handler = FilePathHandler(ROOT_DIR,'Copernicus')
	time_method = CopUVTimeList.time_list_from_minutes
	ID = 'med-cmcc-cur-an-fc-h'

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	@classmethod
	def get_dataset(cls,ID = 'med-cmcc-cur-an-fc-h'):
		username = 'pchamberlain'
		password = 'xixhyg-hebju7-jeBmaf'
		cas_url = 'https://cmems-cas.cls.fr/cas/login'
		session = setup_session(cas_url, username, password)
		session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
		url = cls.base_html+ID
		return open_url(url, session=session)


	@classmethod
	def get_sal_temp_profiles(cls,lat,lon,start_date,end_date):
		lon_idx = cls.lons.find_nearest(lon,idx=True)
		lat_idx = cls.lats.find_nearest(lat,idx=True)
		depth_idx = cls.depth.find_nearest(-700,idx=True)
		time_start_idx = cls.time.find_nearest(start_date,idx=True)
		time_end_idx = cls.time.find_nearest(end_date,idx=True)

		data_list = []
		for var,ID in [('so','med-cmcc-sal-an-fc-h'),('thetao','med-cmcc-tem-an-fc-h')]:
			profile = cls.dataset[var].data[0][time_start_idx:time_end_idx,:depth_idx,lat_idx,lon_idx]
			data_list.append(profile.mean(axis=0).flatten())

		fig, ax1 = plt.subplots()
		color = 'tab:red'
		ax1.set_xlabel('Salinity (psu)', color=color)
		ax1.set_ylabel('Depth (m)')
		ax1.plot(data_list[0], depth[:depth_idx], color=color)
		ax1.tick_params(axis='x', labelcolor=color)

		ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis

		color = 'tab:blue'
		ax2.set_xlabel(r'$\theta_0\ (c)$', color=color)  # we already handled the x-label with ax1
		ax2.plot(data_list[1], depth[:depth_idx], color=color)
		ax2.tick_params(axis='x', labelcolor=color)

		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		gsw.p_from_z(depth,lat)
		density = gsw.density.sigma0(data_list[0],data_list[1])
		fig1, ax1 = plt.subplots()
		ax1.plot(density,depth[:depth_idx])
		plt.xlabel(r'$\sigma_0\ (kg\ m^{-3})$')
		plt.ylabel('depth (m)')
		return (fig,fig1)

	@classmethod
	def get_dataset_shape(cls):
		lllat = min(cls.dataset['lat'][:])
		urlat = max(cls.dataset['lat'][:])
		lllon = min(cls.dataset['lon'][:])
		urlon = max(cls.dataset['lon'][:])
		ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
		return ocean_shape

	@classmethod
	def get_dimensions(cls,urlon,lllon,urlat,lllat,max_depth,dataset):
		time_since = datetime.datetime.strptime(dataset['time'].attributes['units'],'minutes since %Y-%m-%d %H:%M:%S')
		CopUVTimeList.set_ref_date(time_since)
		time = CopUVTimeList.time_list_from_minutes(dataset['time'][:].data.tolist())
		lats = LatList(dataset['lat'][:].data.tolist())
		lons = LonList(dataset['lon'][:].data.tolist())
		depths = DepthList([-x for x in dataset['depth'][:].data.tolist()])
		depth_idx = depths.find_nearest(max_depth,idx=True)
		depths=depths[:depth_idx]
		lllon_idx = lons.find_nearest(lllon,idx=True)
		urlon_idx = lons.find_nearest(urlon,idx=True)
		lllat_idx = lats.find_nearest(lllat,idx=True)
		urlat_idx = lats.find_nearest(urlat,idx=True)
		lons = lons[lllon_idx:urlon_idx]
		lats = lats[lllat_idx:urlat_idx]
		units = dataset['uo'].units
		return (time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units)

	@classmethod
	def download_and_save(cls):
		idx_list = cls.dataset_time.return_time_list()
		k = 0
		while k < len(idx_list)-1:
			print(k)
			k_filename = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'+str(k))
			if os.path.isfile(k_filename):
				k +=1
				continue
			try:
				u_holder = cls.dataset['uo'].data[0][cls.idx_list[k]:cls.idx_list[k+1]
				,:(len(cls.depth))
				,cls.lower_lat_idx:cls.higher_lat_idx
				,cls.lower_lon_idx:cls.higher_lon_idx]
				v_holder = cls.dataset['vo'].data[0][cls.idx_list[k]:cls.idx_list[k+1]
				,:(len(depth))
				,cls.lower_lat_idx:cls.higher_lat_idx
				,cls.lower_lon_idx:cls.higher_lon_idx]
				with open(k_filename, 'wb') as f:
					pickle.dump({'u':u_holder,'v':v_holder, 'time':cls.dataset['time'].data[idx_list[k]:idx_list[k+1]].tolist()},f)
				k +=1
			except:
				print('Index ',k,' encountered an error and did not save. Trying again')
				continue

class CreteCopernicus(Copernicus):
	urlon = 28.5
	lllon = 22.5
	urlat = 38
	lllat = 33
	max_depth = -2000
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Crete'
	PlotClass = CreteCartopy
	dataset = Copernicus.get_dataset()
	dataset_time,lats,lons,depth,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units = Copernicus.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

