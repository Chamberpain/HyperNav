from HyperNav.Utilities.Data.HYCOM import HYCOMMonterey,HYCOMSouthernCalifornia,HYCOMHawaii,HYCOMPuertoRico
from HyperNav.Utilities.Data.CopernicusMed import CreteCopernicus
from HyperNav.Utilities.Data.CopernicusGlobal import TahitiCopernicus,BermudaCopernicus,CanaryCopernicus

from GeneralUtilities.Data.Filepath.instance import FilePathHandler,ComputePathHandler
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import gc
import geopy
import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from copy import deepcopy
from HyperNav.Utilities.Compute.RunParcels import create_prediction,ParticleList,ParticleDataset
import os
import pickle
import shutil

class ControlParticleList(ParticleList):

	def __init__(self,*args,profile,run,**kwargs):
		super().__init__(*args,**kwargs)

class ControlBase():
	calc_days = 6
	profile_num = 61
	def __init__(self,run,start_point,start_time,*args,**kwargs):
		self.run = run
		self.profile = 0
		self.time = []
		self.loc = []
		self.depth = []
		self.load()
		if not self.loc:
			pickle_name = self.make_pickle_filename(self.run,self.profile)
			with open(pickle_name, "wb") as f:
			    pickle.dump([start_time,start_point,0], f)
			self.loc.append(start_point)
			self.time.append(start_time)
		try:
			self.dist = self.calculate_distance()
		except ValueError:
			self.dist = [None]

	def calculate_distance(self):
		return [geopy.distance.GreatCircleDistance(self.loc[0],x).km for x in self.loc]

	def load(self):
		for profile in range(self.profile_num):
				pickle_name = self.make_pickle_filename(self.run,profile)
				try:
					with open(pickle_name, "rb") as f:
						data = pickle.load(f)
				except FileNotFoundError:
					break
				start_time,start_point,drift_depth = data
				self.time.append(start_time)
				self.loc.append(start_point)
				self.depth.append(drift_depth)
				self.profile = profile

	def plot_trajectory(self):
		lat_list = [x.latitude for x in self.loc]
		lon_list = [x.longitude for x in self.loc]

		uv_class = self.uv_class.load(self.time[0],self.time[0]+datetime.timedelta(days=2))
		fig = plt.figure(figsize=(12,12))
		ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		uv_class.PlotClass.urcrnrlat=max(lat_list)+0.2
		uv_class.PlotClass.llcrnrlat=min(lat_list)-0.2
		uv_class.PlotClass.urcrnrlon=max(lon_list)+0.2
		uv_class.PlotClass.llcrnrlon=min(lon_list)-0.2
		X,Y,ax = uv_class.plot(ax=ax)
		ax.plot(lon_list,lat_list,color='black')
		pbc = ax.scatter(lon_list,lat_list,c=self.depth,zorder=20)
		ax.scatter(lon_list[0],lat_list[0],c='black',marker='*',s=500,zorder=21)
		ax.scatter(lon_list[0],lat_list[0],c='pink',marker='*',s=350,zorder=21)
		ax.scatter(lon_list[-1],lat_list[-1],c='black',marker='s',s=400,zorder=21)
		ax.scatter(lon_list[-1],lat_list[-1],c='pink',marker='s',s=300,zorder=21)
		plt.colorbar(mappable=pbc,label='Drift Depth')
		plt.title('Start time = '+str(self.time[0]))

	@classmethod
	def make_zarr_filename(cls,run,profile_num,days,drift_depth):
		filename = '{}_run_{}_profile_{}_days_{}_drift'.format(run,profile_num,days,drift_depth)
		return cls.compute_handler.tmp_file(filename+'.zarr')

	@classmethod
	def make_pickle_filename(cls,run,profile_num):
		filename = '{}_run_{}_profile'.format(run,profile_num)
		return cls.file_handler.tmp_file(filename+'.pickle')

	def make_prediction(self):
		start_time = self.time[-1]
		start_point = self.loc[-1]
		uv_instance = self.uv_class.load(start_time-datetime.timedelta(days=2),start_time+datetime.timedelta(days=7))
		surface_time = 5400
		vertical_speed = 0.076
		for days in range(1,self.calc_days):
			data,dimensions = uv_instance.return_parcels_uv(start_time-datetime.timedelta(days=1),start_time+datetime.timedelta(days=7),start_point)
			date_end = start_time + datetime.timedelta(days=days)
			total_cycle_time = 3600*24*days
			for drift_depth in [50,100,200,300,400,500,600,700]:
				filename = self.make_zarr_filename(self.run,self.profile,days,drift_depth)
				if not os.path.isfile(filename):
					argo_cfg = {
					'lat': start_point.latitude, 'lon': start_point.longitude,
					'time': start_time.timestamp(), 'end_time': date_end.timestamp(), 'depth': 10, 'min_depth': 10, 
					'drift_depth': abs(drift_depth),'max_depth': abs(drift_depth),
					'surface_time': surface_time, 'total_cycle_time': total_cycle_time,
								'vertical_speed': vertical_speed,
								}
					prediction = create_prediction(argo_cfg,data,dimensions,filename)
					gc.collect(generation=2)

	def make_pickle(self):
		try:
			self.make_prediction()
		except AssertionError:
			print('There was an assertion error in make_prediction')
			return
		pl = ParticleList()
		for days in range(1,self.calc_days):
			for drift_depth in [50,100,200,300,400,500,600,700]:
				filename = self.make_zarr_filename(self.run,self.profile,days,drift_depth)
				nc = ParticleDataset(filename)
				pl.append(nc)
		# try:
		start_time,start_point,drift_depth = pl.closest_to_point(self.loc[0])
		# except ValueError:
		# 	for days in range(1,self.calc_days):
		# 		for drift_depth in [50,100,200,300,400,500,600,700]:
		# 			filename = self.make_zarr_filename(self.run,self.profile,days,drift_depth)
		# 			try:
		# 				shutil.rmtree(filename)
		# 			except FileNotFoundError:
		# 				continue			
		# 	gc.collect(generation=2)
		# 	print('There was')
		# 	return		
		pickle_name = self.make_pickle_filename(self.run,self.profile)
		with open(pickle_name, "wb") as f:
		    pickle.dump([start_time,start_point,drift_depth], f)
		    print('saving pickle file at')
		    print(pickle_name)
		self.time.append(start_time)
		self.loc.append(start_point)
		self.depth.append(drift_depth)
		for days in range(1,self.calc_days):
			for drift_depth in [50,100,200,300,400,500,600,700]:
				filename = self.make_zarr_filename(self.run,self.profile,days,drift_depth)
				try:
					shutil.rmtree(filename)
				except FileNotFoundError:
					continue
		gc.collect(generation=2)
		return

	def maintain_location(self):
		for profile in range(self.profile+1,self.profile_num):
			self.profile = profile
			pl = self.make_pickle()
			pickle_name = self.make_pickle_filename(self.run,self.profile)
			# try:
			with open(pickle_name, "rb") as f:
				data = pickle.load(f)
			# except FileNotFoundError:
			# 	break
			start_time,start_point,drift_depth = data
			print('new start point is ')
			print(start_point)
			print('new start time is ')
			print(start_time)
			print('drift depth is ')
			print(drift_depth)
			self.time.append(start_time)
			self.loc.append(start_point)
			self.depth.append(drift_depth)

class ControlMonterey(ControlBase):
	uv_class = HYCOMMonterey
	file_handler = FilePathHandler(ROOT_DIR,'MontereyControl')
	compute_handler = ComputePathHandler(ROOT_DIR,'MontereyControl')

class ControlSoCal(ControlBase):
	uv_class = HYCOMSouthernCalifornia
	file_handler = FilePathHandler(ROOT_DIR,'SoCalControl')
	compute_handler = ComputePathHandler(ROOT_DIR,'SoCalControl')

class ControlHawaii(ControlBase):
	uv_class = HYCOMHawaii
	file_handler = FilePathHandler(ROOT_DIR,'HawaiiControl')
	compute_handler = ComputePathHandler(ROOT_DIR,'HawaiiControl')

class ControlPR(ControlBase):
	uv_class = HYCOMPuertoRico
	file_handler = FilePathHandler(ROOT_DIR,'PRControl')
	compute_handler = ComputePathHandler(ROOT_DIR,'PRControl')

class ControlCrete(ControlBase):
	uv_class = CreteCopernicus
	file_handler = FilePathHandler(ROOT_DIR,'CreteControl')
	compute_handler = ComputePathHandler(ROOT_DIR,'CreteControl')

class ControlTahiti(ControlBase):
	uv_class = TahitiCopernicus
	file_handler = FilePathHandler(ROOT_DIR,'TahitiControl')
	compute_handler = ComputePathHandler(ROOT_DIR,'TahitiControl')

class ControlBermuda(ControlBase):
	uv_class = BermudaCopernicus
	file_handler = FilePathHandler(ROOT_DIR,'BermudaControl')
	compute_handler = ComputePathHandler(ROOT_DIR,'BermudaControl')

class ControlCanary(ControlBase):
	uv_class = CanaryCopernicus
	file_handler = FilePathHandler(ROOT_DIR,'CanaryControl')
	compute_handler = ComputePathHandler(ROOT_DIR,'CanaryControl')