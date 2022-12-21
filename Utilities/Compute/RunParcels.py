from parcels import DiffusionUniformKh, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4,AdvectionDiffusionM1, plotTrajectoriesFile, ErrorCode
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
import numpy as np
from HyperNav.Utilities.Compute.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from datetime import timedelta
import datetime
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from geopy import distance
from GeneralUtilities.Compute.Depth.depth_utilities import PACIOOS as Depth
from parcels import ParcelsRandom as random
from GeneralUtilities.Compute.list import TimeList
import os
import geopy
from GeneralUtilities.Compute.list import LatList,LonList,TimeList

file_handler = FilePathHandler(ROOT_DIR,'RunParcels')


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

	def interpolated_drift_stats(self):
		total_cycle_phase = []
		total_distance = []
		total_speed = []
		for pd in self:
			cycle_phase,distance,speed = pd.return_interpolated_drift_stats()
			total_cycle_phase += cycle_phase
			total_distance += distance
			total_speed += speed
		label_list = []
		speed_list = []
		distance_list = []
		for label in np.unique(total_cycle_phase):
			np.where(np.array(total_cycle_phase)==label)
			label_list.append(label)
			speed_list.append(np.array(total_speed)[np.where(np.array(total_cycle_phase)==label)[0]].mean())
			distance_list.append(np.array(total_distance)[np.where(np.array(total_cycle_phase)==label)[0]].mean())

	def closest_to_point(self,point):
		closest_list = []
		days_list = []
		drift_list = []
		new_pos_list = []
		for ncfid in self: 
			max_days = ncfid.time_from_start()[-1]
			lat_center,lon_center,lat_std,lon_std = ncfid.get_cloud_center(max_days)
			closest_list.append(geopy.distance.GreatCircleDistance(geopy.Point(lat_center,lon_center),point))
			new_pos_list.append(geopy.Point(lat_center,lon_center))
			days_list.append(ncfid.datetime_index()[-1])
			drift_list.append(ncfid['z'][:].max())
		idx = closest_list.index(min(closest_list))
		time = days_list[idx]
		new_pos = new_pos_list[idx]
		drift = drift_list[idx]
		return (time,new_pos,drift)

class ParticleDataset(Dataset):

	def time_from_start(self):
		datetime_list = self.datetime_index()
		return [datetime_list[x] - datetime_list[0] for x in range(len(datetime_list))]

	def datetime_index(self):
		return TimeList([datetime.datetime.fromtimestamp(x) for x in self['time'][:][0,:].tolist()])

	def time_idx(self,timedelta):
		time_list = self.datetime_index()
		return time_list.find_nearest(time_list[0]+timedelta,idx=True)

	def total_geopy_coords(self):
		lats,lons = zip(self.total_coords())
		return [geopy.Point(x,y) for x,y in zip(lats[0].tolist()[0],lons[0].tolist()[0])]

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


	def return_interpolated_drift_stats(self):
		cycle_phase_list = np.array(self.variables['cycle_phase'][0,:]).astype('object')
		cycle_phase_list[cycle_phase_list==0] = 'shear'
		cycle_phase_list[cycle_phase_list==1] = 'drift'
		cycle_phase_list[cycle_phase_list==2] = 'shear'
		cycle_phase_list[cycle_phase_list==3] = 'shear'
		cycle_phase_list[cycle_phase_list==4] = 'surface'


		time_list = self.variables['time'][0,:]
		lat_list = self.variables['lat'][0,:]
		lon_list = self.variables['lon'][0,:]
		idx_list = np.where(np.roll(cycle_phase_list,1)!=cycle_phase_list)[0]
		cycle_phase = []
		start_pos = []
		end_pos = []
		start_time = []
		end_time = []
		distance = []
		speed = []
		for k in range(len(idx_list)-1):
			cycle_phase.append(cycle_phase_list[idx_list[k]])
			start_pos.append(geopy.Point(lat_list[idx_list[k]],lon_list[idx_list[k]]))
			end_pos.append(geopy.Point(lat_list[idx_list[k+1]],lon_list[idx_list[k+1]]))
			start_time.append(datetime.datetime.fromtimestamp(time_list[idx_list[k]]))
			end_time.append(datetime.datetime.fromtimestamp(time_list[idx_list[k+1]]))
			distance.append(geopy.distance.GreatCircleDistance(start_pos[-1],end_pos[-1]).km)
			speed.append(distance[-1]*1000/(end_time[-1]-start_time[-1]).total_seconds())
		return (cycle_phase,distance,speed)



def DeleteParticle(particle, fieldset, time):
    particle.delete()

def create_prediction(float_pos_dict,uv,dimensions,filename,n_particles=500,output_time_step=datetime.timedelta(minutes=15)):

	fieldset = FieldSet.from_data(uv, dimensions,transpose=False)
	fieldset.mindepth = dimensions['depth'][0]
	K_bar = 0.000000000025
	fieldset.add_constant('Kh_meridional',K_bar)
	fieldset.add_constant('Kh_zonal',K_bar)
	particles = dict(lat=float_pos_dict['lat'] + np.random.normal(scale=.05, size=n_particles),
					 lon=float_pos_dict['lon'] + np.random.normal(scale=.05, size=n_particles),
					 time=np.array([float_pos_dict['time']] * n_particles),
					 depth=np.array([float_pos_dict['depth']] * n_particles),
					 min_depth=np.array([float_pos_dict['min_depth'] if 'min_depth' in float_pos_dict.keys() else 10] * n_particles, dtype=np.int32),
					 drift_depth=np.array([float_pos_dict['drift_depth'] if 'drift_depth' in float_pos_dict.keys() else 500] * n_particles, dtype=np.int32),
					 vertical_speed=np.array([float_pos_dict['vertical_speed'] if 'vertical_speed' in float_pos_dict.keys() else 0.076] * n_particles, dtype=np.float32),
					 surface_time=np.array([float_pos_dict['surface_time'] if 'surface_time' in float_pos_dict.keys() else 2 * 3600] * n_particles, dtype=np.int32),
					 cycle_time=np.array([float_pos_dict['total_cycle_time'] if 'total_cycle_time' in float_pos_dict.keys() else 2 * 86400] * n_particles, dtype=np.float32),
					 max_depth=np.array([float_pos_dict['max_depth'] if 'max_depth' in float_pos_dict.keys() else 500] * n_particles, dtype=np.float32),
					 )
	particles['cycle_time'] -= \
	(particles['drift_depth'] / particles['vertical_speed']) \
	+ particles['surface_time'] + \
	abs(particles['max_depth']-particles['drift_depth'])/particles['vertical_speed'] + \
	abs(particles['max_depth'])*particles['vertical_speed']

	particle_set = ParticleSet.from_list(fieldset, pclass=ArgoParticle, **particles)
	kernels = ArgoVerticalMovement + particle_set.Kernel(AdvectionRK4)
	output_file = particle_set.ParticleFile(name=filename,
		outputdt=output_time_step)
	particle_set.execute(kernels,
							  runtime=datetime.timedelta(seconds=(float_pos_dict['end_time']-float_pos_dict['time'])),
						  dt=datetime.timedelta(minutes=3),
						  output_file=output_file,
						  recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
	output_file.export()
	output_file.close()


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






