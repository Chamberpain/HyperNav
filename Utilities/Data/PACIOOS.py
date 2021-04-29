import datetime
from pydap.client import open_url
import numpy as np
from GeneralUtilities.Compute.list import find_nearest
from parcels import DiffusionUniformKh, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4,AdvectionDiffusionM1, plotTrajectoriesFile
from HyperNav.Utilities.Compute.run_parcels import ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50,ArgoParticle
from HyperNav.Utilities.Data.float_position import return_float_pos_dict
from GeneralUtilities.Data.depth.depth_utilities import PACIOOS as Depth
from TransitionMatrix.Utilities.Plot.plot_utils import cartopy_setup
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler



file_handler = FilePathHandler(ROOT_DIR,'PACIOOS')

def add_list(coord_half):
    holder = [coord_half + dummy for dummy in np.random.normal(scale=.1,size=particle_num)]
    return holder

def get_test_particles(float_pos_dict,start_time):
    return ParticleSet.from_list(fieldset,
                                 pclass=ArgoParticle,
                                 lat=np.array(add_list(float_pos_dict['lat'])),
                                 lon=np.array(add_list(float_pos_dict['lon'])),
                                 time=[start_time]*particle_num,
                                 depth=[10]*particle_num
                                 )
particle_num = 500

class DatasetOpenAndParse(object):
	base_html = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap/'
	def __init__(self,float_pos_dict):
		plot_pad = 2
		self.float_pos_dict = float_pos_dict
		self.dataset = open_url(self.base_html+self.ID)
		#open dataset using erdap server
		time_since = datetime.datetime.strptime(self.dataset['time'].attributes['time_origin'],'%d-%b-%Y %H:%M:%S')

		self.time = list(self.dataset['time'][:])
		time_in_datetime = [time_since+datetime.timedelta(seconds=x) for x in self.time]
		lats = list(self.dataset['latitude'][:])
		lons = self.dataset['longitude'][:].data
		lons[lons>180] = lons[lons>180]-360
		lons = list(lons)

		self.higher_lon_idx = lons.index(find_nearest(lons,float_pos_dict['lon']+plot_pad))
		self.lower_lon_idx = lons.index(find_nearest(lons,float_pos_dict['lon']-plot_pad))
		self.lons = lons[self.lower_lon_idx:self.higher_lon_idx]

		self.higher_lat_idx = lats.index(find_nearest(lats,float_pos_dict['lat']+plot_pad))
		self.lower_lat_idx = lats.index(find_nearest(lats,float_pos_dict['lat']-plot_pad))
		self.lats = lats[self.lower_lat_idx:self.higher_lat_idx]

		attribute_dict = self.dataset.attributes['NC_GLOBAL']
		self.time_end = datetime.datetime.strptime(attribute_dict['time_coverage_end'],'%Y-%m-%dT%H:%M:%SZ')
		#define necessary variables from self describing netcdf 

		todays_date = float_pos_dict['datetime'].date()
		todays_date = datetime.datetime.fromordinal(todays_date.toordinal())
		date_array = np.array([todays_date+datetime.timedelta(hours=x) for x in self.hours_list])
		time_diff_list = [abs(x.total_seconds()) for x in (date_array-float_pos_dict['datetime'])]
		closest_idx = time_diff_list.index(min(time_diff_list))
		closest_datetime = todays_date+datetime.timedelta(hours=self.hours_list[closest_idx])
		self.time_idx = time_in_datetime.index(closest_datetime)

class ReturnUV(DatasetOpenAndParse):
	hours_list = np.arange(0,25,3).tolist()
	ID = 'roms_hiig'
	def __init__(self,float_pos_dict):
		super().__init__(float_pos_dict)
		end_time_idx = self.time_idx+len(self.hours_list)*5

		U = self.dataset['u'][self.time_idx:end_time_idx
		,:
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx]
		V = self.dataset['v'][self.time_idx:end_time_idx
		,:
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx]
		W = np.zeros(U.data[0].shape)
		self.data = {'U':U.data[0],'V':V.data[0],'W':W}
		
		self.dimensions = {'time':self.time[self.time_idx:end_time_idx],
		'depth':list(self.dataset['depth'][:]),
		'lat':self.lats,
		'lon':self.lons,}

class ReturnWaves(DatasetOpenAndParse):
	hours_list = np.arange(0,25,1).tolist()
	ID = 'ww3_hawaii'
	def __init__(self,float_pos_dict):
		super().__init__(float_pos_dict)
		end_time_idx = self.time_idx+len(self.hours_list)*5
		self.waves = self.dataset['whgt'][self.time_idx:end_time_idx
		,:
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx].data[0]


class ReturnWeather(DatasetOpenAndParse):
	hours_list = np.arange(0,25,1).tolist()
	ID = 'wrf_hi'
	def __init__(self,float_pos_dict):
		super().__init__(float_pos_dict)
		end_time_idx = self.time_idx+len(self.hours_list)*5
		self.U,self.time,self.lats,self.lons = self.dataset['Uwind'][self.time_idx:end_time_idx
		,:
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx].data


		self.V = self.dataset['Vwind'][self.time_idx:end_time_idx
		,:
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx].data[0]

		self.rain = self.dataset['rain'][self.time_idx:end_time_idx
		,:
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx].data[0]

class ParticleDataset(Dataset):

	def time_idx(self,timedelta):
		time_list = self.variables['time'][:][0,:].tolist()
		time_list = [datetime.timedelta(seconds=(x-time_list[0])) for x in time_list]
		return time_list.index(timedelta)

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


def create_prediction(float_pos_dict,vert_move,drift_depth):
	uv = ReturnUV(float_pos_dict)
	fieldset = FieldSet.from_data(uv.data, uv.dimensions,transpose=False)
	fieldset.mindepth = uv.dimensions['depth'][0]
	K_bar = 0.000000000025
	fieldset.add_constant('Kh_meridional',K_bar)
	fieldset.add_constant('Kh_zonal',K_bar)
    kona_pos = (19.6400,-155.9969)
    lahaina_pos = (20.8700,-156.68)

	aground_list = []
	lahaina_list = []
	kona_list = []
	float_move_list = []

	testParticles = get_test_particles(float_pos_dict,uv.dimensions['time'][0])
	kernels = vert_move + testParticles.Kernel(AdvectionRK4)
	dt = 15 #15 minute timestep
	output_file = testParticles.ParticleFile(name=file_handler.tmp_file('Uniform_out.nc'),
		outputdt=datetime.timedelta(minutes=dt))
	testParticles.execute(kernels,
	                      runtime=datetime.timedelta(days=3),
	                      dt=datetime.timedelta(minutes=dt),
	                      output_file=output_file,)
	output_file.export()
	output_file.close()

def historical_prediction():
	from HyperNav.Utilities.Data.previous_traj_parse import gps_42_file,mission_42_file,NavisParse
	parser = NavisParse(gps_42_file,mission_42_file,start_idx = 3)
	diff_list = []
	while looper:
		try: 
			float_pos_dict,dummy = parser.increment_profile()
			idx = parser.index
			create_prediction(float_pos_dict,ArgoVerticalMovement700,700)
			nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
			for time in [datetime.timedelta(days=x) for x in [1,2,3]]:
				(ax,fig) = cartopy_setup(nc,float_pos_dict)
				lats,lons = nc.get_cloud_snapshot(time)
				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(time)
				plt.scatter(lons.tolist(),lats.tolist(),c='m',s=2,zorder=5,label='Prediction')
				plt.scatter(float_pos_dict['lon'],float_pos_dict['lat'],marker='x',c='k',linewidth=6,s=250,zorder=6,label='Location')
				plt.scatter(lon_center,lat_center,marker='x',c='b',linewidth=6,s=250,zorder=6,label='Mean Prediction')
				float_pos_dict_future,dummy = parser.increment_profile()
				plt.scatter(float_pos_dict_future['lon'],float_pos_dict_future['lat'],marker='o',c='k',linewidth=6,s=250,zorder=6,label=('Actual '+str(time)+' Day Later'))
				plt.legend()
				plt.savefig(file_handler.out_file('historical_day_'+str(idx)+'_prediction_'+str(time)))
				plt.close()
				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(time)

				float_future_pos = (float_pos_dict_future['lat'],float_pos_dict_future['lon'])
				mean_prediction_pos = (lat_center,lon_center)
				diff_list.append((GreatCircleDistance(float_future_pos,mean_prediction_pos),time))
			parser.index = idx
		except KeyError:
			looper = False
		distance_error_list,days = zip(diff_list)
		mean_list = []
		std_list = []
		for day in np.sort(np.unique(days)):
			tmp_array = np.array(distance_error)[np.array(days)==day]
			mean_list.append(tmp_array.mean())
			std_list.append(tmp_array.std())
		plt.plot(np.sort(np.unique(days)),mean_list)
		plt.fill_between(days,np.array(mean_list)-np.array(std_list),np.array(mean_list)+np.array(std_list),color='red',alpha=0.2)



def multi_depth_prediction():
	float_pos_dict = return_float_pos_dict()
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


def cartopy_setup(nc,float_pos_dict):
	import cartopy.crs as ccrs
	import cartopy.feature as cfeature
	pad = 1
	urlat = nc['lat'][:].max()
	lllat = nc['lat'][:].min()
	urlon = nc['lon'][:].max()
	lllon = nc['lon'][:].min()
	lon_0 = 0
	center_lat = 20.8
	center_lon = -157.2
	llcrnrlon=(lllon-pad)
	llcrnrlat=(lllat-pad)
	urcrnrlon=(urlon+pad)
	urcrnrlat=(urlat+pad)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
	ax.scatter(float_pos_dict['lon'],float_pos_dict['lat'])
	ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())
	ax.add_feature(cfeature.LAND)
	ax.add_feature(cfeature.COASTLINE)
	ax.set_aspect('auto')
	gl = ax.gridlines(draw_labels=True)
	gl.xlabels_top = False
	gl.ylabels_right = False	
	return ax,fig

def plot_total_prediction(depth_level):
	float_pos_dict = return_float_pos_dict()
	nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
	depth = Depth()

	(ax,fig) = cartopy_setup(nc,float_pos_dict)
	XX1,YY1 = np.meshgrid(depth.x,depth.y)
	plt.contour(XX1,YY1,depth.z,[-1*depth_level],colors=('k',),linewidths=(7,),zorder=4,label='Drift Depth Contour')
	plt.contourf(XX1,YY1,np.ma.masked_greater(depth.z/1000.,0),zorder=3)
	plt.colorbar(label = 'Depth (km)')

	particle_color = np.zeros(nc['lon'].shape)
	for k in range(particle_color.shape[1]):
		particle_color[:,k]=k

	plt.scatter(nc['lon'][:],nc['lat'][:],c=particle_color,cmap='PuRd',s=2,zorder=5)
	plt.scatter(float_pos_dict['lon'],float_pos_dict['lat'],marker='x',c='k',linewidth=10,s=500,zorder=6,label='Location')
	plt.legend()
	plt.savefig(file_handler.out_file('bathy_plot_total_'+str(depth_level)))
	plt.close()

def plot_snapshot_prediction(depth_level):
	float_pos_dict = return_float_pos_dict()
	nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
	depth = Depth()

	for time in [datetime.timedelta(days=x) for x in [1,2,3]]:
		(ax,fig) = cartopy_setup(nc,float_pos_dict)
		XX1,YY1 = np.meshgrid(depth.x,depth.y)
		plt.contour(XX1,YY1,depth.z,[-1*depth_level],colors=('k',),linewidths=(7,),zorder=4,label='Drift Depth Contour')
		plt.contourf(XX1,YY1,np.ma.masked_greater(depth.z/1000.,0),zorder=3)
		plt.colorbar(label = 'Depth (km)')


		lats,lons = nc.get_cloud_snapshot(time)
		lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(time)
		plt.scatter(lons.tolist(),lats.tolist(),c='m',s=2,zorder=5,label='Prediction')
		plt.scatter(float_pos_dict['lon'],float_pos_dict['lat'],marker='x',c='k',linewidth=6,s=250,zorder=6,label='Location')
		plt.scatter(lon_center,lat_center,marker='x',c='b',linewidth=6,s=250,zorder=6,label='Mean Prediction')

		plt.legend()

		plt.savefig(file_handler.out_file('bathy_plot_days-'+str(time.days)+'_depth-'+str(int(depth_level))))
		plt.close()	


pword = 'n5vkJ?lw\EmdidlJ'
server_ip = '204.197.4.164'
	
