from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.Data.UVBase import Base,UVTimeList
from GeneralUtilities.Compute.list import LatList, LonList, DepthList, flat_list
from urllib.error import HTTPError
from GeneralUtilities.Plot.Cartopy.regional_plot import KonaCartopy, CanaryCartopy, BermudaCartopy, TahitiCartopy
from socket import timeout
import matplotlib.pyplot as plt
import gsw
import os
import pickle
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
from GeneralUtilities.Plot.Cartopy.regional_plot import CreteCartopy
from GeneralUtilities.Compute.Depth.depth_utilities import ETopo1Depth
import datetime
import gsw
import shapely.geometry
import numpy as np
from HyperNav.Utilities.Data.CopernicusMed import CopUVTimeList

class CopernicusGlobal(Base):
	facecolor = 'orange'
	dataset_description = 'Copernicus'
	base_html = 'https://nrt.cmems-du.eu/thredds/dodsC/'
	time_step = datetime.timedelta(hours=1)
	hours_list = np.arange(0,25,1).tolist()
	DepthClass = ETopo1Depth
	file_handler = FilePathHandler(ROOT_DIR,'Copernicus')
	time_method = CopUVTimeList.time_list_from_hours
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	@classmethod
	def get_dataset(cls,ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'):
		username = 'pchamberlain'
		password = 'xixhyg-hebju7-jeBmaf'
		cas_url = 'https://cmems-cas.cls.fr/cas/login'
		session = setup_session(cas_url, username, password)
		session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
		url = cls.base_html+ID
		return open_url(url, session=session)


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
		time_since = datetime.datetime.strptime(dataset['time'].attributes['units'],'hours since %Y-%m-%d')
		time = cls.time_method(dataset['time'][:].data.tolist(),time_since)
		lats = LatList(dataset['latitude'][:].data.tolist())
		lons = LonList(dataset['longitude'][:].data.tolist())
		depths = DepthList([-x for x in dataset['depth'][:].data.tolist()])
		depth_idx = depths.find_nearest(max_depth,idx=True)
		depths=depths[:depth_idx]
		depths[0] = 0
		depths[-1] = -700
		lllon_idx = lons.find_nearest(lllon,idx=True)
		urlon_idx = lons.find_nearest(urlon,idx=True)
		lllat_idx = lats.find_nearest(lllat,idx=True)
		urlat_idx = lats.find_nearest(urlat,idx=True)
		lons = lons[lllon_idx:urlon_idx]
		lats = lats[lllat_idx:urlat_idx]
		units = dataset['uo'].units
		return (time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,time_since)

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
				u_holder = cls.dataset['uo'].data[0][idx_list[k]:idx_list[k+1]
				,:(len(cls.depths))
				,cls.lllat_idx:cls.urlat_idx
				,cls.lllon_idx:cls.urlon_idx]
				v_holder = cls.dataset['vo'].data[0][idx_list[k]:idx_list[k+1]
				,:(len(cls.depths))
				,cls.lllat_idx:cls.urlat_idx
				,cls.lllon_idx:cls.urlon_idx]
				with open(k_filename, 'wb') as f:
					pickle.dump({'u':u_holder,'v':v_holder, 'time':cls.dataset['time'].data[idx_list[k]:idx_list[k+1]].tolist()},f)
				k +=1
			except:
				print('Index ',k,' encountered an error and did not save. Trying again')
				continue

class HawaiiCopernicus(CopernicusGlobal):
	urlat = 22
	lllat = 16
	lllon = -159
	urlon = -154
	max_depth = -800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Hawaii'
	PlotClass = KonaCartopy
	dataset = CopernicusGlobal.get_dataset()
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class TahitiCopernicus(CopernicusGlobal):
	urlat = -15
	lllat = -21
	lllon = -152.5
	urlon = -147.0
	max_depth = -800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Tahiti'
	PlotClass = TahitiCartopy
	dataset = CopernicusGlobal.get_dataset()
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)


class BermudaCopernicus(CopernicusGlobal):
	urlat = 34.5
	lllat = 29.5
	lllon = -67
	urlon = -62
	max_depth = -800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Bermuda'
	PlotClass = BermudaCartopy
	dataset = CopernicusGlobal.get_dataset()
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class CanaryCopernicus(CopernicusGlobal):
	urlat = 30.0
	lllat = 25.0
	lllon = -19.0
	urlon = -14.0
	max_depth = -800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Canary'
	PlotClass = CanaryCartopy
	dataset = CopernicusGlobal.get_dataset()
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)
