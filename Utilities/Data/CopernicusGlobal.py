from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.Data.UVBase import Base,UVTimeList
from GeneralUtilities.Compute.list import LatList, LonList, DepthList, flat_list
from urllib.error import HTTPError
from GeneralUtilities.Plot.Cartopy.regional_plot import KonaCartopy, CanaryCartopy, BermudaCartopy, TahitiCartopy, CCSCartopy, PuertoRicoCartopy
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
import copernicusmarine
# from HyperNav.Utilities.Data.CopernicusMed import CopUVTimeList

class CopUVTimeList(UVTimeList):
	def return_time_list(self):
		return list(range(len(self))[::40])+[len(self)-1] #-1 because of pythons crazy list indexing

def nanosecond_convert(time_list,ref_date):
	time = [np.timedelta64(x, 'ns')+ ref_date for x in time_list]
	time = [((x - ref_date)
             / np.timedelta64(1, 's')) for x in time]
	time = [datetime.datetime.utcfromtimestamp(x) for x in time]
	return time

class CopernicusGlobal(Base):
	facecolor = 'green'
	dataset_description = 'GOPAF'
	time_step = datetime.timedelta(hours=6)
	hours_list = np.arange(0,25,6).tolist()
	DepthClass = ETopo1Depth
	file_handler = FilePathHandler(ROOT_DIR,'Copernicus')
	time_method = nanosecond_convert
	copernicusmarine.login(username='pchamberlain', password='xixhyg-hebju7-jeBmaf', overwrite_configuration_file=True)
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	@classmethod
	def get_dataset(cls,urlat,lllat,urlon,lllon,max_depth,ID,start_date = '2022-06-01',end_date = (datetime.date.today()+datetime.timedelta(days=15)).isoformat()):
		dataset = copernicusmarine.open_dataset(
		    dataset_id = ID,
		    minimum_longitude = lllon,
		    maximum_longitude = urlon,
		    minimum_latitude = lllat,
		    maximum_latitude = urlat,
		    start_datetime = start_date,
		    end_datetime = end_date,
		    variables = ['uo', 'vo'],
            minimum_depth=0,
            maximum_depth=800,
		)
		return dataset


	@classmethod
	def get_dataset_shape(cls):
		lllat = min(cls.dataset['latitude'].data)
		urlat = max(cls.dataset['latitude'].data)
		lllon = min(cls.dataset['latitude'].data)
		urlon = max(cls.dataset['latitude'].data)
		ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
		return ocean_shape

	@classmethod
	def get_dimensions(cls,urlon,lllon,urlat,lllat,max_depth,dataset):
		time = [((x - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's')) for x in dataset['time'].data]
		time = [datetime.datetime.utcfromtimestamp(x) for x in time]
		time = CopUVTimeList(time)
		lats = LatList(dataset['latitude'][:].data.tolist())
		lons = LonList(dataset['longitude'][:].data.tolist())
		depths = DepthList([-x for x in dataset['depth'][:].data.tolist()])
		depth_idx = -1
		depths[0] = 0
		depths[-1] = -800
		units = dataset['uo'].units
		return (time,lats,lons,depths,0,-1,0,-1,units,np.datetime64('1970-01-01T00:00:00'))

	@classmethod
	def download_and_save(cls):
		idx_list = cls.dataset_time.return_time_list()
		k = 0
		while k < len(idx_list)-1:
			temp_dataset = cls.get_dataset(cls.urlat,
				cls.lllat,
				cls.urlon,
				cls.lllon,
				cls.max_depth,
				cls.ID,
				cls.dataset_time[idx_list[k]].isoformat(),
				(cls.dataset_time[idx_list[k+1]]-cls.time_step).isoformat()
				)
			print(k)
			k_filename = cls.make_k_filename(k)
			print(k_filename)
			if os.path.isfile(k_filename):
				k +=1
				continue
			try:
				u_holder = temp_dataset['uo'].data[:
				,:
				,:
				,:]
				v_holder = temp_dataset['vo'].data[:
				,:
				,:
				,:]
				with open(k_filename, 'wb') as f:
					pickle.dump({'u':u_holder,'v':v_holder, 'time':temp_dataset['time'].data.tolist()},f)
				f.close()
				k +=1
			except:
				print('Index ',k,' encountered an error and did not save. Trying again')
				continue

	@classmethod
	def download_recent(cls):
		cls.delete_latest()
		idx_list = cls.dataset_time.return_time_list()
		k = len(idx_list)-10
		while k < len(idx_list)-1:
			temp_dataset = cls.get_dataset(cls.urlat,
				cls.lllat,
				cls.urlon,
				cls.lllon,
				cls.max_depth,
				cls.ID,
				cls.dataset_time[idx_list[k]].isoformat(),
				(cls.dataset_time[idx_list[k+1]]-cls.time_step).isoformat()
				)
			print(k)
			k_filename = cls.make_k_filename(k)
			print(k_filename)
			if os.path.isfile(k_filename):
				k +=1
				continue
			try:
				u_holder = temp_dataset['uo'].data[:
				,:
				,:
				,:]
				v_holder = temp_dataset['vo'].data[:
				,:
				,:
				,:]
				with open(k_filename, 'wb') as f:
					pickle.dump({'u':u_holder,'v':v_holder, 'time':temp_dataset['time'].data.tolist()},f)
				f.close()
				k +=1
			except:
				print('Index ',k,' encountered an error and did not save. Trying again')
				continue


class SoCalCopernicus(CopernicusGlobal):
	location='SoCal'
	facecolor = 'Pink'
	urlat = 35
	lllat = 30
	lllon = -122
	urlon = -116.5
	max_depth = 800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'SouthernCalifornia'
	PlotClass = CCSCartopy
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
	dataset = CopernicusGlobal.get_dataset(urlat,lllat,urlon,lllon,max_depth,ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class MontereyCopernicus(CopernicusGlobal):
	location='Monterey'
	facecolor = 'Pink'
	urlat = 39
	lllat = 34
	lllon = -126
	urlon = -121.5
	max_depth = -700
	PlotClass = CCSCartopy
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
	dataset = CopernicusGlobal.get_dataset(urlat,lllat,urlon,lllon,max_depth,ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class PuertoRicoCopernicus(CopernicusGlobal):
	location = 'PuertoRico'
	facecolor = 'yellow'
	urlon = -65
	lllon = -68.5 
	urlat = 22.5
	lllat = 16
	max_depth = -700
	PlotClass = PuertoRicoCartopy
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
	dataset = CopernicusGlobal.get_dataset(urlat,lllat,urlon,lllon,max_depth,ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class TahitiCopernicus(CopernicusGlobal):
	urlat = -15
	lllat = -21
	lllon = -152.5
	urlon = -147.0
	max_depth = 800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Tahiti'
	PlotClass = TahitiCartopy
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
	dataset = CopernicusGlobal.get_dataset(urlat,lllat,urlon,lllon,max_depth,ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)



class HawaiiCopernicus(CopernicusGlobal):
	urlat = 22
	lllat = 16
	lllon = -158
	urlon = -154
	max_depth = 800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Hawaii'
	PlotClass = KonaCartopy
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
	dataset = CopernicusGlobal.get_dataset(urlat,lllat,urlon,lllon,max_depth,ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class HawaiiOffshoreCopernicus(CopernicusGlobal):
	urlat = 17.5
	lllat = 14.5
	lllon = -158
	urlon = -154
	max_depth = 800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'HawaiiOffshore'
	PlotClass = KonaCartopy
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
	dataset = CopernicusGlobal.get_dataset(urlat,lllat,urlon,lllon,max_depth,ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)



class BermudaCopernicus(CopernicusGlobal):
	urlat = 34.5
	lllat = 29.5
	lllon = -67
	urlon = -62
	max_depth = 800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Bermuda'
	PlotClass = BermudaCartopy
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
	dataset = CopernicusGlobal.get_dataset(urlat,lllat,urlon,lllon,max_depth,ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)

class CanaryCopernicus(CopernicusGlobal):
	urlat = 30.0
	lllat = 25.0
	lllon = -19.0
	urlon = -14.0
	max_depth = 800
	ocean_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon([[lllon, urlat], [urlon, urlat], [urlon, lllat], [lllon, lllat], [lllon, urlat]])])	
	location = 'Canary'
	PlotClass = CanaryCartopy
	ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i'
	dataset = CopernicusGlobal.get_dataset(urlat,lllat,urlon,lllon,max_depth,ID)
	dataset_time,lats,lons,depths,lllon_idx,urlon_idx,lllat_idx,urlat_idx,units,ref_date = CopernicusGlobal.get_dimensions(urlon,lllon,urlat,lllat,max_depth,dataset)
