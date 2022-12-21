from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
import numpy as np
import os
from GeneralUtilities.Data.Lagrangian.Argo.array_class import ArgoArray
from HyperNav.Utilities.Data.Instrument import ArgoFloat
from HyperNav.Utilities.Compute.RunParcels import ParticleDataset,create_prediction
import geopy
from GeneralUtilities.Data.pickle_utilities import load,save
import gc
from parcels.tools.statuscodes import OutOfBoundsError, OutOfTimeError
import xarray
from geopy.distance import GreatCircleDistance
from geopy import Point
import pandas as pd
from GeneralUtilities.Compute.list import LatList,LonList
from GeneralUtilities.Data.Filepath.instance import FilePathHandler


def dx_dy_distance(point1,point2):

	lat1 = point1.latitude
	lon1 = point1.longitude
	lat2 = point2.latitude
	lon2 = point2.longitude
	dy = np.sign(lat2-lat1)*GreatCircleDistance(Point(lat1,lon1),Point(lat2,lon1)).km
	dx1 = np.sign(lon2-lon1)*GreatCircleDistance(Point(lat1,lon1),Point(lat1,lon2)).km
	dx2 = np.sign(lon2-lon1)*GreatCircleDistance(Point(lat2,lon1),Point(lat2,lon2)).km
	dx = (dx1+dx2)/2
	return (dy,dx)


class ErrorCalcBase:
	def __init__(self):
		array = self.DataArray.compile()
		array = array.get_regional_subset(self.CurrentData.ocean_shape,self.CurrentData.location)
		dict_list = []
		for drifter in array.values():
			drifter = self.DrifterClass(drifter)
			dict_list += drifter.return_float_pos_dict_list(self.CurrentData.ocean_shape)
		results = []
		for k,(start_time, end_time, prof_dict) in  enumerate(dict_list):
			if start_time-datetime.timedelta(days=5)<min(self.CurrentData.dataset_time):
				continue
			filename = self.file_handler.tmp_file(self.ErrorDescription+str(k))
			try:
				results.append(load(filename))
			except FileNotFoundError:
				uv_class = HYCOMHawaii.load(start_time-datetime.timedelta(days=5),end_time+datetime.timedelta(days=5))
				data,dimensions = uv_class.return_parcels_uv(start_time-datetime.timedelta(days=5),end_time+datetime.timedelta(days=5))
				try:
					create_prediction(prof_dict,data,dimensions,self.file_handler.tmp_file('Uniform_out.zarr'),out_of_bounds_recovery=True)
				except OutOfBoundsError:
					print('I am out of bounds')
					print('lat ,',prof_dict['lat'])
					print('lon ,',prof_dict['lon'])
					data = ()
					save(filename,data)
					continue
				except OutOfTimeError:
					print('I am out of time')
					data = ()
					save(filename,data)
					continue
				except IndexError:
					print('There was an index problem')
					data = ()
					save(filename,data)
					continue
				nc = ParticleDataset(xarray.open_zarr(self.file_handler.tmp_file('Uniform_out.zarr')))
				try:
					lat,lon,dum,dum = nc.get_cloud_center(end_time-start_time)
				except TypeError:
					print('I encountered a type error. Continuing')
					continue
				data = (start_time,geopy.Point(prof_dict['target_lat'],prof_dict['target_lon']),geopy.Point(lat,lon))
				save(filename,data)
				results.append(data)
				gc.collect(generation=2)
		self.residuals = []
		for time,target_pos,pos in results:
			dx,dy = dx_dy_distance(target_pos,pos)
			total_dist = GreatCircleDistance(target_pos,pos).km
			if total_dist>100:
				print(target_pos)
				print(pos)
			self.residuals.append((dx,dy,total_dist,time,target_pos.latitude,target_pos.longitude))

	def error_timeseries(self):
		df = pd.DataFrame({'dx':dx,'dy':dy,'dist':total_dist,'time':time,'lat':lat,'lon':lon})
		df = df.set_index('time').resample('5d').mean().dropna()

class ArgoHawaiiError(ErrorCalcBase):
	CurrentData = HYCOMHawaii
	DrifterClass = ArgoFloat
	DataArray = ArgoArray
	ErrorDescription = 'Argo_Drifter_Error_'
	file_handler = FilePathHandler(ROOT_DIR,'HypernavHawaiiFieldDeployment')
