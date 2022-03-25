from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from HyperNav.Utilities.Data.UVBase import Base,UVTimeList
from GeneralUtilities.Compute.list import LatList, LonList, DepthList, flat_list
from urllib.error import HTTPError
from socket import timeout
import os
import pickle
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
from GeneralUtilities.Plot.Cartopy.regional_plot import CreteCartopy
from GeneralUtilities.Data.depth.depth_utilities import ETopo1Depth
import datetime


class CopUVTimeList(UVTimeList):
	def return_time_list(self):
		return list(range(len(self))[::3])+[len(self)-1] #-1 because of pythons crazy list indexing

class Copernicus(Base):
	urlon = 28.5
	lllon = 22.5
	urlat = 38
	lllat = 33
	dataset_description = 'Copernicus'
	location = 'Crete'
	base_html = 'https://nrt.cmems-du.eu/thredds/dodsC/'
	PlotClass = CreteCartopy
	DepthClass = ETopo1Depth
	file_handler = FilePathHandler(ROOT_DIR,'Copernicus')
	time_method = CopUVTimeList.time_list_from_minutes

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		dataset = open_url(self.base_html+self.ID)
		time_since = datetime.datetime.strptime(dataset['time'].attributes['units'],'minutes since %Y-%m-%d %H:%M:%S')
		self.time.set_ref_date(time_since)

	@classmethod
	def get_dataset(cls,ID = 'med-cmcc-cur-an-fc-h'):
		username = 'pchamberlain'
		password = 'xixhyg-hebju7-jeBmaf'
		cas_url = 'https://cmems-cas.cls.fr/cas/login'
		session = setup_session(cas_url, username, password)
		session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
		url = cls.base_html+ID
		return open_url(url, session=session)

	# def get_temp():
	# 	self.get_dataset(ID = 'med-cmcc-sal-an-fc-h')


	# def get_sal():




	@classmethod
	def get_dimensions(cls):
		dataset = cls.get_dataset()
		time_since = datetime.datetime.strptime(dataset['time'].attributes['units'],'minutes since %Y-%m-%d %H:%M:%S')
		CopUVTimeList.set_ref_date(time_since)
		time = CopUVTimeList.time_list_from_minutes(dataset['time'][:].data.tolist())
		lats = LatList(dataset['lat'][:].data.tolist())
		lons = LonList(dataset['lon'][:].data.tolist())
		depths = DepthList([-x for x in dataset['depth'][:].data.tolist()])
		depth_idx = depths.find_nearest(-2000,idx=True)
		depths=depths[:depth_idx]
		lllon_idx = lons.find_nearest(cls.lllon,idx=True)
		urlon_idx = lons.find_nearest(cls.urlon,idx=True)
		lllat_idx = lats.find_nearest(cls.lllat,idx=True)
		urlat_idx = lats.find_nearest(cls.urlat,idx=True)
		lons = lons[lllon_idx:urlon_idx]
		lats = lats[lllat_idx:urlat_idx]
		units = dataset['uo'].units
		return (time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units)

	@classmethod
	def download_and_save(cls):
		dataset = cls.get_dataset()
		time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,units = cls.get_dimensions()
		idx_list = time.return_time_list()
		k = 0
		while k < len(idx_list)-1:
			print(k)
			k_filename = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'+str(k))
			if os.path.isfile(k_filename):
				k +=1
				continue
			try:
				u_holder = dataset['uo'].data[0][idx_list[k]:idx_list[k+1]
				,:(len(depth))
				,lower_lat_idx:higher_lat_idx
				,lower_lon_idx:higher_lon_idx]
				v_holder = dataset['vo'].data[0][idx_list[k]:idx_list[k+1]
				,:(len(depth))
				,lower_lat_idx:higher_lat_idx
				,lower_lon_idx:higher_lon_idx]
				with open(k_filename, 'wb') as f:
					pickle.dump({'u':u_holder,'v':v_holder, 'time':dataset['time'].data[idx_list[k]:idx_list[k+1]].tolist()},f)
				k +=1
			except:
				print('Index ',k,' encountered an error and did not save. Trying again')
				continue