from parcels import DiffusionUniformKh, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4,AdvectionDiffusionM1, plotTrajectoriesFile, ErrorCode
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
import numpy as np
from HyperNav.Utilities.Compute.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from datetime import timedelta
import datetime
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from geopy import distance
from GeneralUtilities.Data.depth.depth_utilities import PACIOOS as Depth
from parcels import ParcelsRandom as random
from GeneralUtilities.Compute.list import TimeList
import os
import geopy
import h5py
from GeneralUtilities.Compute.list import LatList,LonList,TimeList
from HyperNav.Data.__init__ import ROOT_DIR as DATA_DIR


file_handler = FilePathHandler(ROOT_DIR,'RunParcels')

class ClearSky():
	def __init__(self,month):
		self.aot = h5py.File(os.path.join(DATA_DIR,'aot869_'+month+'_Hawaii.h5'))
		self.percent_aot = h5py.File(os.path.join(DATA_DIR,'percent_aot869_le_0.1_'+month+'_Hawaii.h5'))
		self.percent_clear = h5py.File(os.path.join(DATA_DIR,'percent_clear_days_'+month+'_Hawaii.h5'))

		self.lat = LatList(self.aot['lat'][:,0].tolist())
		self.lon = LonList(self.aot['lon'][0,:].tolist())

	def return_data(self,data,std,lat,lon):
		lat_idx = self.lat.find_nearest(lat,idx=True)
		lon_idx = self.lon.find_nearest(lon,idx=True)
		return np.random.normal(loc=data[lat_idx,lon_idx],scale=std[lat_idx,lon_idx])

	def return_aot_percent(self,lat,lon):
		return self.return_data(self.percent_aot['percent AOT869<=0.1 (mean)'],self.percent_aot['percent AOT869<=0.1 (std)'],lat,lon)
			
	def return_aot(self,lat,lon):
		return self.return_data(self.aot['AOT869 (mean)'],self.aot['AOT869 (std)'],lat,lon)

	def return_clear_sky(self,lat,lon):
		return self.return_data(self.percent_clear['percent clear days (mean)'],self.percent_clear['percent clear days (std)'],lat,lon)

class ParticleList(list):

	def get_cloud_snapshot(self,timedelta):
		lats = []
		lons = []
		for pd in self:
			lats_holder,lons_holder = pd.get_cloud_snapshot(timedelta)
			lats += lats_holder.data.tolist()
			lons += lons_holder.data.tolist()
		return (lats,lons)

	def get_time(self,timedelta):
		time_out = []
		for pd in self:
			time_list = TimeList.time_list_from_seconds(pd.variables['time'][:][0,:].tolist())
			time = time_list[0]+timedelta
			time_out += [time]*pd.variables['time'].shape[0]
		return time_out

	def plot_density(self,timedelta,bins,ax):
		lats,lons = self.get_cloud_snapshot(timedelta)
		H,x,y = np.histogram2d(lons,lats,bins=bins,density=True)
		H = np.ma.masked_equal(H.T,0)
		XX,YY = np.meshgrid(x[:-1],y[:-1],indexing='xy')
		ax.contourf(XX,YY,np.log(H),vmin=0,vmax=3)
		return ax



class ParticleDataset(Dataset):

	def time_idx(self,timedelta):
		time_list = TimeList.time_list_from_seconds(self.variables['time'][:][0,:].tolist())
		return time_list.find_nearest(time_list[0]+timedelta,idx=True)

	def total_coords(self):
		return (self.variables['lat'][:],self.variables['lon'][:])

	def get_cloud_snapshot(self,timedelta):
		time_idx = self.time_idx(timedelta)
		lats,lons = self.total_coords()
		return (lats[:,time_idx],lons[:,time_idx])

	def get_cloud_center(self,timedelta):
		lats,lons = self.get_cloud_snapshot(timedelta)
		lat_center = lats.mean()
		lat_std = lats.std()
		lon_center = lons.mean()
		lon_std = lons.std()
		return (lat_center,lon_center,lat_std,lon_std)

	def get_depth_snapshot(self,timedelta):
		depth = Depth()
		lats,lons = self.get_cloud_snapshot(timedelta)
		return [depth.return_z(x) for x in zip(lats,lons)]

	def within_bounds(self,position):
		bnds_list = []
		for ii in [1,2,3]:
			lats,lons = self.get_cloud_snapshot(datetime.timedelta(days=ii))
			dist_list = [geopy.distance.GreatCircleDistance(float_pos,position).nm>50 for float_pos in zip(lats,lons)]
			bnds_list.append(np.sum(dist_list)/len(lats)*100)
		return bnds_list

	def dist_list(self,float_dict):
		float_pos = (float_dict['lat'],float_dict['lon'])
		dist_list = []
		for ii in [1,2,3]:
			lat,lon,dummy,dummy = self.get_cloud_center(datetime.timedelta(days=ii))
			dist_list.append(geopy.distance.GreatCircleDistance(float_pos,(lat,lon)).nm)
		return dist_list

	def percentage_aground(self,depth_level):
		depth = Depth()
		(ax,fig) = cartopy_setup(nc,float_pos_dict)
		XX1,YY1 = np.meshgrid(depth.x,depth.y)
		cs = plt.contour(XX1,YY1,depth.z,[-1*depth_level-0.1*depth_level])
		plt.close()
		paths = cs.collections[0].get_paths()
		lat,lon = self.total_coords()
		cloud_mean_tuple = (lat.mean(),lon.mean())
		problem_idxs = []
		for k,path in enumerate(paths):
			mean_path_tuple = (path.vertices[:,1].mean(),path.vertices[:,0].mean()) #path outputs are in x,y format
			cloud_to_path_dist = geopy.distance.GreatCircleDistance(mean_path_tuple,cloud_mean_tuple).nm
			if cloud_to_path_dist>90: #computationally efficient way of not computing impossible paths
				continue
			truth_dummy = path.contains_points(list(zip(lon.flatten(),lat.flatten())))
			row_idx,dummy = np.where(truth_dummy.reshape(lat.shape))
			problem_idxs+=row_idx.tolist()
		problem_idxs = np.unique(problem_idxs)
		return (np.unique(problem_idxs).shape[0]/lat.shape[0])*100 #number of problem floats/number of total floats


def DeleteParticle(particle, fieldset, time):
    particle.delete()

class UVPrediction():

	def __init__(self,float_pos_dict,uv,dimensions,*args,**kwargs):
		self.float_pos_dict = float_pos_dict
		self.uv = uv
		self.dimensions = dimensions

	def create_prediction(self,n_particles=500,filename = 'Uniform_out.nc',output_time_step=datetime.timedelta(minutes=15)):
		fieldset = FieldSet.from_data(self.uv, self.dimensions,transpose=False)
		fieldset.mindepth = self.dimensions['depth'][0]
		K_bar = 0.000000000025
		fieldset.add_constant('Kh_meridional',K_bar)
		fieldset.add_constant('Kh_zonal',K_bar)
		particles = dict(lat=self.float_pos_dict['lat'] + np.random.normal(scale=.05, size=n_particles),
						 lon=self.float_pos_dict['lon'] + np.random.normal(scale=.05, size=n_particles),
						 time=np.array([self.float_pos_dict['time']] * n_particles),
						 depth=np.array([self.float_pos_dict['depth']] * n_particles),
						 min_depth=np.array([self.float_pos_dict['min_depth'] if 'min_depth' in self.float_pos_dict.keys() else 10] * n_particles, dtype=np.int32),
						 drift_depth=np.array([self.float_pos_dict['drift_depth'] if 'drift_depth' in self.float_pos_dict.keys() else 500] * n_particles, dtype=np.int32),
						 vertical_speed=np.array([self.float_pos_dict['vertical_speed'] if 'vertical_speed' in self.float_pos_dict.keys() else 0.076] * n_particles, dtype=np.float32),
						 surface_time=np.array([self.float_pos_dict['surface_time'] if 'surface_time' in self.float_pos_dict.keys() else 2 * 3600] * n_particles, dtype=np.int32),
						 cycle_time=np.array([self.float_pos_dict['total_cycle_time'] if 'total_cycle_time' in self.float_pos_dict.keys() else 2 * 86400] * n_particles, dtype=np.float32),
						 max_depth=np.array([self.float_pos_dict['max_depth'] if 'max_depth' in self.float_pos_dict.keys() else 500] * n_particles, dtype=np.float32),
						 )
		particles['cycle_time'] -= \
		(particles['drift_depth'] / particles['vertical_speed']) + particles['surface_time'] + \
		abs(particles['max_depth']+particles['drift_depth'])*particles['vertical_speed'] + \
		abs(particles['max_depth'])*particles['vertical_speed']

		particle_set = ParticleSet.from_list(fieldset, pclass=ArgoParticle, **particles)
		kernels = ArgoVerticalMovement + particle_set.Kernel(AdvectionRK4)
		output_file = particle_set.ParticleFile(name=file_handler.tmp_file(filename),
			outputdt=output_time_step)
		particle_set.execute(kernels,
							  runtime=datetime.timedelta(seconds=(self.float_pos_dict['end_time']-self.float_pos_dict['time'])),
							  dt=datetime.timedelta(minutes=3),
							  output_file=output_file,
							  recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
		output_file.export()
		output_file.close()

	def calculate_prediction(self):
		predictions = []
		self.create_prediction(vert_move_dict[depth_level],days=days)
		nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
		nc['cycle_age'][0,:].data
		holder = nc['time'][0,:]
		assert ([x-holder[0] for x in holder][:10] == nc['cycle_age'][0,:].data[:10]).all()
		#time must be passing the same for the float
		for k,time in enumerate([datetime.timedelta(days=x) for x in np.arange(.2,days,.1)]):
			try: 
				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(time)
			except ValueError:
				continue
			date_string = (self.float_pos_dict['datetime']+time).isoformat()
			id_string = int(str(self.float_pos_dict['ID'])+'0'+str(depth_level)+'0'+str(k))
			dummy_dict = {"prediction_id":id_string,
			"datetime":date_string,
			"lat":float(lat_center),
			"lon":float(lon_center),
			"uncertainty":0,
			"model":'HYCOM'+'_'+str(depth_level)}
			predictions.append(dummy_dict)
		return predictions

	def upload_single_depth_prediction(self,depth_level):
		SiteAPI.delete_by_model('HYCOM'+'_'+str(depth_level),self.float_pos_dict['ID'])
		predictions = self.calculate_prediction(depth_level,days=1)
		SiteAPI.upload_prediction([x for x in predictions if x['model']=='HYCOM'+'_'+str(depth_level)],self.float_pos_dict['ID'])

	def upload_multi_depth_prediction(self):
		for depth_level in vert_move_dict.keys():
			self.upload_single_depth_prediction(depth_level)

	def plot_multi_depth_prediction(self):
		color_dict = {50:'red',100:'purple',200:'blue',300:'teal',
		400:'pink',500:'tan',600:'orange',700:'yellow'}
		self.create_prediction(vert_move_dict[50])
		nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
		XX,YY,ax = HypernavCartopy(nc,self.float_pos_dict,lon_grid=self.uv.lons,lat_grid=self.uv.lats,pad=-0.5).get_map()
		depth = Depth()
		XX1,YY1 = np.meshgrid(depth.x,depth.y)
		plt.contour(XX1,YY1,depth.z,[-1*self.float_pos_dict['park_pressure']],colors=('k',),linewidths=(4,),zorder=4,label='Drift Depth Contour')
		plt.contourf(XX1,YY1,np.ma.masked_greater(depth.z/1000.,0),zorder=3,cmap=plt.get_cmap('Greys'))
		plt.colorbar(label='Depth (km)')
		plt.scatter(self.float_pos_dict['lon'],self.float_pos_dict['lat'],marker='x',c='k',linewidth=6,s=250,zorder=6,label='Location')
		for k in range(particle_num):
			lats = nc['lat'][k,:]
			lons = nc['lon'][k,:]
			plt.plot(lons,lats,linewidth=2,zorder=10)
		plt.title('Float '+str(self.float_pos_dict['ID'])+' at '+datetime.datetime.now().isoformat())
		savefile =str(self.float_pos_dict['ID'])+'_Multidepth_'+str(self.float_pos_dict['profile'])		
		plt.savefig(file_handler.out_file(savefile))
		plt.close()	

class ArgoParticle(JITParticle):
	# Phase of cycle: init_descend=0, drift=1, profile_descend=2, profile_ascend=3, transmit=4
	cycle_phase = Variable('cycle_phase', dtype=np.int32, initial=0.)
	cycle_age = Variable('cycle_age', dtype=np.float32, initial=0.)
	surface_age = Variable('surface_age', dtype=np.float32, initial=0.)
	profile_idx = Variable('profile_idx', dtype=np.float32, initial=0.)
	#temp = Variable('temp', dtype=np.float32, initial=np.nan)  # if fieldset has temperature
	drift_depth = Variable('drift_depth', dtype=np.int32,  to_write=False)  # drifting depth in m
	min_depth = Variable('min_depth', dtype=np.int32, to_write=False)       # shallowest depth in m
	max_depth = Variable('max_depth', dtype=np.int32, to_write=False)     # profile depth in m
	vertical_speed = Variable('vertical_speed', dtype=np.float32, to_write=False)  # sink and rise speed in m/s  (average speed of profile 0054.21171)
	surface_time = Variable('surface_time', dtype=np.int32, to_write=False)        # surface time in seconds
	cycle_time = Variable('cycle_time', dtype=np.float32, to_write=False)          # total time of cycle in seconds






