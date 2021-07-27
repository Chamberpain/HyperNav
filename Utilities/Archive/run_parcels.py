from parcels import DiffusionUniformKh, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4,AdvectionDiffusionM1, plotTrajectoriesFile
import numpy as np
from datetime import timedelta
from operator import attrgetter
from HyperNav.Utilities.Data.data_parse import raw_base_file, processed_base_file
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from geopy import distance
from GeneralUtilities.Data.depth.depth_utilities import PACIOOS as Depth
from parcels import ParcelsRandom as random
import os


def add_list(coord_half):
	holder = [coord_half + dummy for dummy in np.random.normal(scale=.1,size=particle_num)]
	return holder

def get_test_particles(fieldset,float_pos_dict,start_time):
	return ParticleSet.from_list(fieldset,
								 pclass=ArgoParticle,
								 lat=np.array(add_list(float_pos_dict['lat'])),
								 lon=np.array(add_list(float_pos_dict['lon'])),
								 time=[start_time]*particle_num,
								 depth=[10]*particle_num
								 )
particle_num = 500


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

class UVPrediction():

	def __init__(self,float_pos_dict,uv=False,*args,**kwargs):
		if not uv:
			uv = ReturnHYCOMUV(float_pos_dict,*args,**kwargs)
		self.float_pos_dict = float_pos_dict
		self.uv = uv

	def create_prediction(self,vert_move,days=3):
		fieldset = FieldSet.from_data(self.uv.data, self.uv.dimensions,transpose=False)
		fieldset.mindepth = self.uv.dimensions['depth'][0]
		K_bar = 0.000000000025
		fieldset.add_constant('Kh_meridional',K_bar)
		fieldset.add_constant('Kh_zonal',K_bar)
		testParticles = get_test_particles(fieldset,self.float_pos_dict,self.uv.dimensions['time'][0])
		kernels = vert_move + testParticles.Kernel(AdvectionRK4)
		dt = 15 #15 minute timestep
		output_file = testParticles.ParticleFile(name=file_handler.tmp_file('Uniform_out.nc'),
			outputdt=datetime.timedelta(minutes=dt))
		testParticles.execute(kernels,
							  runtime=datetime.timedelta(days=days),
							  dt=datetime.timedelta(minutes=dt),
							  output_file=output_file,)
		output_file.export()
		output_file.close()

	def calculate_prediction(self,depth_level,days=3.):
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


	def bonus():
		for vert_move,drift_depth in zip([ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50],[700,600,500,400,300,200,100,50]):
			percent_aground,percent_lahaina_in_bounds,percent_kona_in_bounds,float_move = create_prediction(float_pos_dict,vert_move,drift_depth)
			plot_total_prediction(drift_depth)
			plot_snapshot_prediction(drift_depth)
			nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
			percent_aground = nc.percentage_aground(depth_level = drift_depth)
			percent_lahaina_in_bounds = nc.within_bounds(lahaina_pos)
			percent_kona_in_bounds = nc.within_bounds(kona_pos)
			float_move = nc.dist_list(float_pos_dict)
			aground_list.append((percent_aground,drift_depth))
			lahaina_list.append((percent_lahaina_in_bounds,drift_depth))
			kona_list.append((percent_kona_in_bounds,drift_depth))
			float_move_list.append((float_move,drift_depth))

		plt.bar(depth,aground_percent,width=25)
		plt.ylabel('Grounding Percentage')
		plt.xlabel('Depth')
		plt.savefig(file_handler.out_file('grounding_percentage'))
		plt.close()

		def plot_position(pos_list):
			for percent_outside,depth in pos_list:
				plt.plot([1,2,3],percent_outside,label=(str(depth)+'m Depth'))
			plt.legend()
			plt.xlabel('Days')


		plt.figure()
		plot_position(lahaina_list)
		plt.ylabel('Percent of Floats Outside Operations Area')
		plt.savefig(file_handler.out_file('lahaina_percentage'))
		plt.close()

		plt.figure()
		plot_position(kona_list)
		plt.ylabel('Percent of Floats Outside Operations Area')
		plt.savefig(file_handler.out_file('kona_percentage'))
		plt.close()

		plt.figure()
		plot_position(float_move_list)
		plt.ylabel('Distance from Starting Point Floats Move (nm)')
		plt.savefig(file_handler.out_file('float_dist'))
		plt.close()




def add_list(list_):
	holder = []
	for item in list_:
		holder += [item + dummy for dummy in np.random.normal(scale=.1,size=particle_num)]
	return holder


# Define the new Kernel that mimics Argo vertical movement
def ArgoVerticalMovement700(particle, fieldset, time):
	driftdepth = 700  # maximum depth in m
	vertical_speed = 0.10  # sink and rise speed in m/s
	cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
	surftime = 2 * 3600  # time of deep drift in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_age = 0  # reset cycle_age for next cycle
			particle.cycle_phase = 3

	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		#particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
		if particle.depth <= mindepth:
			particle.depth = mindepth
			#particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0

	if particle.state == ErrorCode.Evaluate:
		particle.cycle_age += particle.dt  # update cycle_age


def ArgoVerticalMovement600(particle, fieldset, time):
	driftdepth = 600  # maximum depth in m
	vertical_speed = 0.10  # sink and rise speed in m/s
	cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
	surftime = 2 * 3600  # time of deep drift in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_age = 0  # reset cycle_age for next cycle
			particle.cycle_phase = 3

	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		#particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
		if particle.depth <= mindepth:
			particle.depth = mindepth
			#particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0

	if particle.state == ErrorCode.Evaluate:
		particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement500(particle, fieldset, time):
	driftdepth = 500  # maximum depth in m
	vertical_speed = 0.10  # sink and rise speed in m/s
	cycletime = 1 * (21*3600-driftdepth/vertical_speed)  # total time of cycle in seconds
	surftime = 2 * 3600  # time of deep drift in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_age = 0  # reset cycle_age for next cycle
			particle.cycle_phase = 3

	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		#particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
		if particle.depth <= mindepth:
			particle.depth = mindepth
			#particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0

def ArgoVerticalMovement400(particle, fieldset, time):
	driftdepth = 400  # maximum depth in m
	vertical_speed = 0.10  # sink and rise speed in m/s
	cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
	surftime = 2 * 3600  # time of deep drift in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_age = 0  # reset cycle_age for next cycle
			particle.cycle_phase = 3

	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		#particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
		if particle.depth <= mindepth:
			particle.depth = mindepth
			#particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0

	if particle.state == ErrorCode.Evaluate:
		particle.cycle_age += particle.dt  # update cycle_age


def ArgoVerticalMovement300(particle, fieldset, time):
	driftdepth = 300  # maximum depth in m
	vertical_speed = 0.10  # sink and rise speed in m/s
	cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
	surftime = 2 * 3600  # time of deep drift in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_age = 0  # reset cycle_age for next cycle
			particle.cycle_phase = 3

	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		#particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
		if particle.depth <= mindepth:
			particle.depth = mindepth
			#particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0

	if particle.state == ErrorCode.Evaluate:
		particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement200(particle, fieldset, time):
	driftdepth = 200  # maximum depth in m
	vertical_speed = 0.10  # sink and rise speed in m/s
	cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
	surftime = 2 * 3600  # time of deep drift in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_age = 0  # reset cycle_age for next cycle
			particle.cycle_phase = 3

	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		#particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
		if particle.depth <= mindepth:
			particle.depth = mindepth
			#particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0

	if particle.state == ErrorCode.Evaluate:
		particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement100(particle, fieldset, time):
	driftdepth = 100  # maximum depth in m
	vertical_speed = 0.1  # sink and rise speed in m/s
	cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
	surftime = 2 * 3600  # time of deep drift in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_age = 0  # reset cycle_age for next cycle
			particle.cycle_phase = 3

	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		#particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
		if particle.depth <= mindepth:
			particle.depth = mindepth
			#particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0

	if particle.state == ErrorCode.Evaluate:
		particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement50(particle, fieldset, time):
	driftdepth = 50  # maximum depth in m
	vertical_speed = 0.1  # sink and rise speed in m/s
	cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
	surftime = 2 * 3600  # time of deep drift in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_age = 0  # reset cycle_age for next cycle
			particle.cycle_phase = 3

	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		#particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
		if particle.depth <= mindepth:
			particle.depth = mindepth
			#particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0

	if particle.state == ErrorCode.Evaluate:
		particle.cycle_age += particle.dt  # update cycle_age


class ArgoParticle(JITParticle):
	# Phase of cycle: init_descend=0, drift=1, profile_descend=2, profile_ascend=3, transmit=4
	cycle_phase = Variable('cycle_phase', dtype=np.int32, initial=0.)
	cycle_age = Variable('cycle_age', dtype=np.float32, initial=0.)
	surf_age = Variable('surf_age', dtype=np.float32, initial=0.)
	profile_idx = Variable('profile_idx', dtype=np.float32, initial=0.)
	#temp = Variable('temp', dtype=np.float32, initial=np.nan)  # if fieldset has temperature

def get_test_particles():
	return ParticleSet.from_list(fieldset,
								 pclass=ArgoParticle,
								 lat=np.array(add_list(lat_list)),
								 lon=np.array(add_list(lon_list)),
								 time=np.zeros(particle_num*len(lat_list)),
								 depth=[10]*particle_num*len(lat_list)
								 )
def run_parcels():
	particle_num = 300
	lat_list = [20.7,20.5,20.9,19.7,19.4,20.0]
	lon_list = [-157.3,-157.3,-157.4,-156.2,-156.1,-156.1]
	depth_700 = 18

	K_bar = 0.000000000025

	dimensions = {'lat': 'lat',
	'lon':'lon',
	'time':'time',
	'depth':'depth'}

	variables = {'U':'u','V':'v'}

	files = os.listdir(raw_base_file)



	for file in files:
		if not file.endswith('.nc'):
			continue
		filenames = {'U':raw_base_file+file,
		'V':raw_base_file+file}

		fieldset = FieldSet.from_netcdf(filenames,variables,dimensions)
		fieldset.add_constant('Kh_meridional',K_bar)
		fieldset.add_constant('Kh_zonal',K_bar)
		testParticles = get_test_particles()
		kernels = testParticles.Kernel(ArgoVerticalMovement)
		dt = 15 #5 minute timestep
		output_file = testParticles.ParticleFile(name=processed_base_file+file.split('.')[0]+"_Uniform_out.nc",
												 outputdt=timedelta(minutes=dt))
		testParticles.execute(kernels,
							  runtime=timedelta(days=28),
							  dt=timedelta(minutes=dt),
							  output_file=output_file,)
		output_file.export()
		output_file.close()


	start_lat_list=[]
	start_lon_list=[]
	end_lat_list=[]
	end_lon_list=[]
	total_depth_flag_list = []
	depth = Depth()
	files = os.listdir(processed_base_file)
	kona_pos = (19.6400,-155.9969)
	lahaina_pos = (20.8700,-156.68)

	for file in files:
		if not file.endswith('.nc'):
			continue
		nc = Dataset(processed_base_file+file)

		float_depth_flag_list = []
		for k in range(nc.variables["lon"][:].shape[0]):
			print(k)
			lats = nc.variables["lat"][:][k,:].data
			lons = nc.variables["lon"][:][k,:].data

			start_lat_list.append(lats[0])
			start_lon_list.append(lons[0])
			end_lat_list.append(lats[-1])
			end_lon_list.append(lons[-1])

			for x in list(zip(lats[::10],lons[::10])):
				print(depth.return_z(x))
				depth_truth = depth.too_shallow(x)
				if depth_truth:
					break
			if depth_truth:
				float_depth_flag_list.append(False)
				print(float_depth_flag_list[-1])
				continue
			float_depth_flag_list.append(True)
			print(float_depth_flag_list[-1])
		total_depth_flag_list += float_depth_flag_list
		nc.close()
		# plotTrajectoriesFile(processed_base_file+"Uniform_out.nc")
		# testParticles.show(field=fieldset.V,depth_level=depth_700,time=0)

	# x = nc.variables["lon"][:][0,:].squeeze()
	# y = nc.variables["lat"][:][0,:].squeeze()
	# z = nc.variables["z"][:][0,:].squeeze()
	# fig = plt.figure(figsize=(13,10))
	# ax = plt.axes(projection='3d')
	# cb = ax.scatter(x, y, z, c=z, s=20, marker="o")
	# ax.set_xlabel("Longitude")
	# ax.set_ylabel("Latitude")
	# ax.set_zlabel("Depth (m)")
	# ax.set_zlim(np.max(z),0)
	# plt.savefig(processed_base_file+'z_prof_example')
	# plt.close()
	files = os.listdir(raw_base_file)
	for file in files:
		if not file.endswith('.nc'):
			continue
		model_nc = Dataset(raw_base_file+file)
		lon_grid = model_nc['lon'][:]
		lat_grid = model_nc['lat'][:]
	XX,YY,m = basemap_setup(lat_grid,lon_grid,'Moby')

	nx = 50
	ny = 50

	lon_bins = np.linspace(lon_grid.min(), lon_grid.max(), nx+1)
	lat_bins = np.linspace(lat_grid.min(), lat_grid.max(), ny+1)

	density, _, _ = np.histogram2d(start_lat_list,start_lon_list, [lat_bins, lon_bins])
	density = np.ma.masked_equal(density,0)
	lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
	xs, ys = m(lon_bins_2d, lat_bins_2d)
	plt.pcolormesh(xs, ys, density)
	plt.colorbar(orientation='horizontal',label='Number of Floats Deployed in Bin')
	plt.scatter(*m(end_lon_list,end_lat_list),s=0.4,c='k',alpha=0.4)
	plt.savefig(processed_base_file+'density_plot')
	plt.close()


	dist_list_kona = [distance.great_circle(kona_pos,x).nm for x in zip(end_lat_list,end_lon_list)]
	dist_list_lahaina = [distance.great_circle(lahaina_pos,x).nm for x in zip(end_lat_list,end_lon_list)]

	kona_dist_flag_list = [(kona<50) for kona in dist_list_kona]
	lahaina_dist_flag_list = [(lahaina<50) for lahaina in dist_list_lahaina]
	total_dist_flag_list = [(kona<50)|(lahaina<50) for kona,lahaina in zip(dist_list_kona,dist_list_lahaina)]

	lats_bin = depth.y.data
	lons_bin = depth.x.data

	result = scipy.stats.binned_statistic_2d(start_lon_list,start_lat_list,total_depth_flag_list,statistic='mean',bins=[lons_bin[400:800],lats_bin[200:600]])
	YY,XX = np.meshgrid(result.y_edge[:-1],result.x_edge[:-1])
	XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
	m.pcolormesh(XX,YY,(1-result.statistic.T)*100)
	plt.colorbar(label='Chance of becoming bathymetric sensor (%)')
	plt.title('Map of mean grounding chance')
	plt.savefig(processed_base_file+'moby_grounding_map')
	plt.close()
	result = scipy.stats.binned_statistic_2d(start_lon_list,start_lat_list,dist_list_kona,statistic='mean',bins=[lons_bin,lats_bin])
	YY,XX = np.meshgrid(result.y_edge[:-1],result.x_edge[:-1])
	XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
	m.pcolormesh(XX,YY,(result.statistic.T))
	plt.colorbar(label='Distance from Kona at end of run (nm)')
	plt.title('Map of mean distance')
	plt.savefig(processed_base_file+'moby_kona_distance_map')
	plt.close()
	mask = np.array(total_depth_flag_list)&np.array(kona_dist_flag_list)
	lons = np.array(start_lon_list)[mask]
	lats = np.array(start_lat_list)[mask]
	XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
	m.scatter(lons,lats,latlon=True)
	plt.title('Deployment locations of successful particles for Kona')
	plt.savefig(processed_base_file+'moby_kona_success map')
	plt.close()
	result = scipy.stats.binned_statistic_2d(start_lon_list,start_lat_list,dist_list_lahaina,statistic='mean',bins=[lons_bin,lats_bin])
	YY,XX = np.meshgrid(result.y_edge[:-1],result.x_edge[:-1])
	XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
	m.pcolormesh(XX,YY,(result.statistic.T))
	plt.colorbar(label='Distance from Lahaina at end of run (nm)')
	plt.title('Map of mean distance')
	plt.savefig(processed_base_file+'moby_lahaina_distance_map')
	plt.close()
	mask = np.array(total_depth_flag_list)&np.array(lahaina_dist_flag_list)
	lons = np.array(start_lon_list)[mask]
	lats = np.array(start_lat_list)[mask]
	XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
	m.scatter(lons,lats,latlon=True)
	plt.title('Deployment locations of successful particles for Lahaina ')
	plt.savefig(processed_base_file+'moby_lahaina_success map')
	plt.close()
	mask = np.array(total_depth_flag_list)&np.array(total_dist_flag_list)
	lons = np.array(start_lon_list)[mask]
	lats = np.array(start_lat_list)[mask]
	XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
	m.scatter(lons,lats,latlon=True)
	plt.title('All successful deployement locations')
	plt.savefig(processed_base_file+'moby_total_success map')
	plt.close()
	result = scipy.stats.binned_statistic_2d(start_lon_list,start_lat_list,mask,statistic='mean',bins=[lons_bin[400:800][::4],lats_bin[200:600][::4]])
	YY,XX = np.meshgrid(result.y_edge[:-1][::4],result.x_edge[:-1][::4])
	XX,YY,m = basemap_setup(result.y_edge[:-1][::4],result.x_edge[:-1][::4],'Moby')
	m.pcolormesh(XX,YY,(1-result.statistic.T)*100)
	plt.colorbar(label='Chance of Success (%)')
	plt.title('Map of successful deployments')
	plt.savefig(processed_base_file+'moby_success_map')
	plt.close()

