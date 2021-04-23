import datetime
from pydap.client import open_url
import numpy as np
from compute_utilities.list_utilities import find_nearest
from parcels import DiffusionUniformKh, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4,AdvectionDiffusionM1, plotTrajectoriesFile
from hypernav.run_parcels import ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50,ArgoParticle
from hypernav.file_download.float_position import return_float_pos_dict
from data_save_utilities.depth.depth_utilities import PACIOOS as Depth
from transition_matrix.makeplots.plot_utils import cartopy_setup
import matplotlib.pyplot as plt

filename = 'Uniform_out.nc'

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


def create_prediction():
	float_pos_dict = return_float_pos_dict()
	uv = ReturnUV(float_pos_dict)
	fieldset = FieldSet.from_data(uv.data, uv.dimensions,transpose=False)
	fieldset.mindepth = uv.dimensions['depth'][0]
	K_bar = 0.000000000025
	fieldset.add_constant('Kh_meridional',K_bar)
	fieldset.add_constant('Kh_zonal',K_bar)

	# kernels = ArgoVerticalMovement + 

	for vert_move,drift_depth in zip([ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50],[700,600,500,400,300,200,100,50]):

		testParticles = get_test_particles(float_pos_dict,uv.dimensions['time'][0])
		kernels = vert_move + testParticles.Kernel(AdvectionRK4)
		dt = 15 #15 minute timestep
		output_file = testParticles.ParticleFile(name=filename,
		                                         outputdt=datetime.timedelta(minutes=dt))
		testParticles.execute(kernels,
		                      runtime=datetime.timedelta(days=3),
		                      dt=datetime.timedelta(minutes=dt),
		                      output_file=output_file,)
		output_file.export()
		output_file.close()
		plot_prediction(drift_depth)


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

def plot_prediction(depth_level):
	from netCDF4 import Dataset
	float_pos_dict = return_float_pos_dict()
	nc = Dataset(filename)
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

	plt.savefig('bathy_plot_'+str(depth_level))
	plt.close()





	pword = 'n5vkJ?lw\EmdidlJ'
	server_ip = '204.197.4.164'
	
