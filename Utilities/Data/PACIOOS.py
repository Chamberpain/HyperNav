import datetime
from pydap.client import open_url
import numpy as np
from GeneralUtilities.Compute.list import find_nearest, TimeList
from parcels import DiffusionUniformKh, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4,AdvectionDiffusionM1, plotTrajectoriesFile
from HyperNav.Utilities.Compute.run_parcels import ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50,ArgoParticle
from HyperNav.Utilities.Data.float_position import return_float_pos_dict,return_float_pos_list
from GeneralUtilities.Data.depth.depth_utilities import PACIOOS as Depth
from GeneralUtilities.Plot.Cartopy.eulerian_plot import HypernavCartopy
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import os
import requests


vert_move_dict={50:ArgoVerticalMovement50,100:ArgoVerticalMovement100,200:ArgoVerticalMovement200,
				300:ArgoVerticalMovement300,400:ArgoVerticalMovement400,500:ArgoVerticalMovement500,
				600:ArgoVerticalMovement600,700:ArgoVerticalMovement700}

file_handler = FilePathHandler(ROOT_DIR,'PACIOOS')

def add_list(coord_half):
	holder = [coord_half + dummy for dummy in np.random.normal(scale=.5,size=particle_num)]
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

class DatasetOpenAndParse(object):
	def __init__(self,float_pos_dict,*args,plot_pad=2,**kwargs):
		self.float_pos_dict = float_pos_dict
		self.dataset = open_url(self.base_html+self.ID)
		#open dataset using erdap server
		time_since = datetime.datetime.strptime(self.dataset['time'].attributes['time_origin'],'%d-%b-%Y %H:%M:%S')
		TimeList.set_ref_date(time_since)
		self.time = TimeList.time_list_from_seconds(self.dataset['time'][:])
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
		#define necessary variables from self describing netcdf 
		attribute_dict = self.dataset.attributes['NC_GLOBAL']
		self.time_end = datetime.datetime.strptime(attribute_dict['time_coverage_end'],'%Y-%m-%dT%H:%M:%SZ')

		todays_date = float_pos_dict['datetime'].date()
		todays_date = datetime.datetime.fromordinal(todays_date.toordinal())
		self.time_idx = self.time.closest_index(todays_date)

class HYCOMDataOpenAndParse(DatasetOpenAndParse):
	base_html = 'https://www.ncei.noaa.gov/erddap/griddap/'
	def __init__(self,float_pos_dict,*args,**kwargs):
		super().__init__(float_pos_dict,*args,**kwargs)

class ReturnHYCOMUV(HYCOMDataOpenAndParse):
	hours_list = np.arange(0,25,3).tolist()
	ID = 'HYCOM_reg6_latest3d'
	def __init__(self,float_pos_dict,*args,days=5,**kwargs):
		super().__init__(float_pos_dict,*args,**kwargs)
		depth_idx = 35
		end_time_idx = self.time_idx+len(self.hours_list)*days
		U = self.dataset['water_u'][self.time_idx:end_time_idx
		,:depth_idx
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx]
		V = self.dataset['water_v'][self.time_idx:end_time_idx
		,:depth_idx
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx]
		W = np.zeros(U.data[0].shape)
		self.data = {'U':U.data[0],'V':V.data[0],'W':W}
		
		self.dimensions = {'time':self.time.seconds_since()[self.time_idx:end_time_idx],
		'depth':list(self.dataset['depth'][:depth_idx]),
		'lat':self.lats,
		'lon':self.lons,}

	def get_new_snapshot(self,float_pos_dict,days=5):
		class BlankUV():
			pass

		holderUV = BlankUV()
		TimeList.set_ref_date(self.time.ref_date)
		time = TimeList.time_list_from_seconds(self.dimensions['time'])
		start_date = float_pos_dict['datetime'].date()
		start_date = datetime.datetime.fromordinal(start_date.toordinal())
		end_date = start_date+datetime.timedelta(days=days)
		start_time_idx = time.closest_index(start_date)
		end_time_idx = time.closest_index(end_date)
		time = time.seconds_since()[start_time_idx:end_time_idx]
		U = self.data['U'][start_time_idx:end_time_idx,:,:,:]
		V = self.data['V'][start_time_idx:end_time_idx,:,:,:]
		W = self.data['W'][start_time_idx:end_time_idx,:,:,:]
		holderUV.data = {'U':U,'V':V,'W':W}
		holderUV.float_pos_dict = float_pos_dict
		holderUV.dimensions = {'time':time,
		'depth':self.dimensions['depth'],
		'lat':self.dimensions['lat'],
		'lon':self.dimensions['lon']}
		return holderUV,

class PACIOOSDataOpenAndParse(DatasetOpenAndParse):
	base_html = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap/'
	def __init__(self,float_pos_dict,*args,**kwargs):
		super().__init__(float_pos_dict,*args,**kwargs)

class ReturnPACIOOSUV(PACIOOSDataOpenAndParse):
	hours_list = np.arange(0,25,3).tolist()
	ID = 'roms_hiig'
	def __init__(self,float_pos_dict,*args,days=5,**kwargs):
		super().__init__(float_pos_dict,*args,**kwargs)
		end_time_idx = self.time_idx+len(self.hours_list)*days

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
		
		self.dimensions = {'time':self.time.seconds_since()[self.time_idx:end_time_idx],
		'depth':list(self.dataset['depth'][:]),
		'lat':self.lats,
		'lon':self.lons,}

class ReturnPACIOOSWaves(PACIOOSDataOpenAndParse):
	hours_list = np.arange(0,25,1).tolist()
	ID = 'ww3_hawaii'
	def __init__(self,float_pos_dict,*args,days=5,**kwargs):
		super().__init__(float_pos_dict,*args,**kwargs)
		end_time_idx = self.time_idx+len(self.hours_list)*days

		self.waves = self.dataset['Thgt'][self.time_idx:end_time_idx
		,:
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx].data[0]


class ReturnPACIOOSWeather(PACIOOSDataOpenAndParse):
	hours_list = np.arange(0,25,1).tolist()
	ID = 'wrf_hi'
	def __init__(self,float_pos_dict,*args,days=5,**kwargs):
		super().__init__(float_pos_dict,*args,**kwargs)
		end_time_idx = self.time_idx+len(self.hours_list)*days
		self.U,time,self.lats,self.lons = self.dataset['Uwind'][self.time_idx:end_time_idx
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx].data
		self.time = TimeList.time_list_from_seconds(time)


		self.V = self.dataset['Vwind'][self.time_idx:end_time_idx
		,self.lower_lat_idx:self.higher_lat_idx
		,self.lower_lon_idx:self.higher_lon_idx].data[0]

		self.rain = self.dataset['rain'][self.time_idx:end_time_idx
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
		predictions = self.calculate_prediction(depth_level)
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

class SiteAPI():
	api_url = 'http://misclab.umeoce.maine.edu/HyperNAV/api/v1/'
	access_token = 'DRehY4yxIFog2o8BZaNk752TqFgN3hHG8yRsB1uGfImwbXmJ8jM1ENQxsvJDCiOZJGNpeo4AL7bKQXVcLWDUV6rSEusEq'

	@staticmethod
	def upload_prediction(predictions,ID):
		response = requests.put(SiteAPI.api_url + 'position_predictions/' + ID,
								headers={'Authorization': SiteAPI.access_token},
								json=predictions)
		if response.status_code == 200:
			print('Success')
		else:
			print(response.status_code, response.reason)	

	@staticmethod
	def get_past_locations(platform_id):
		class PredictionList(list):

			def return_lats(self):
				return [x['lat'] for x in self]

			def return_lons(self):
				return [x['lon'] for x in self]

			def return_time(self):
				date_string_list = [x['datetime'] for x in self]
				return [datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S+00:00') for x in date_string_list]

			def return_depth(self):
				return [x['park_pressure'] for x in self]


		response = requests.get(SiteAPI.api_url + 'meta/' + platform_id,
								params={'limit': 50})
		if response.status_code == 200:
			holder = []
			data = response.json()
			for x in [x for x in data if x['datetime']!=None]:
				try: 
					x['park_pressure']
					holder.append(x)
				except KeyError:
					continue
			return PredictionList(holder[::-1])

		else:
			print(response.status_code, response.reason)

	@staticmethod
	def position_predictions(platform_id):
		response = requests.get(SiteAPI.api_url + 'position_predictions/' + platform_id)
		if response.status_code == 200:
			data = response.json()
			for r in data:
				print(r)
		else:
			print(response.status_code, response.reason)

	@staticmethod
	def delete_by_model(model,platform_id):
		model_dict = {'model': model}
		response = requests.delete(SiteAPI.api_url + 'position_predictions/' + platform_id,
								headers={'Authorization': SiteAPI.access_token},
								params=model_dict)
		if response.status_code == 200:
			print('Success')
		else:
			print(response.status_code, response.reason)


def make_forecast():
	for float_pos_dict in return_float_pos_list():
		uv = UVPrediction(float_pos_dict)
		uv.single_depth_prediction(700)
		uv.multi_depth_prediction()

def historical_prediction():
	from HyperNav.Utilities.Data.previous_traj_parse import gps_42_file,mission_42_file,NavisParse
	parser = NavisParse(gps_42_file,mission_42_file,start_idx = 3)
	diff_list = []
	while looper:
		try: 
			float_pos_dict,dummy = parser.increment_profile()
			idx = parser.index
			create_prediction(float_pos_dict,ArgoVerticalMovement700,700,UVClass=ReturnHYCOMUV)
			nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
			for time in [datetime.timedelta(days=x) for x in [1,2,3]]:
				(ax,fig) = cartopy_setup(nc,float_pos_dict)
				lats,lons = nc.get_cloud_snapshot(time)
				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(time)
				plt.scatter(lons.tolist(),lats.tolist(),c='m',s=2,zorder=5,label='Prediction')
				plt.scatter(float_pos_dict['lon'],float_pos_dict['lat'],marker='x',c='k',linewidth=6,s=250,zorder=6,label='Location')
				plt.scatter(lon_center,lat_center,marker='x',c='b',linewidth=6,s=250,zorder=6,label='Mean Prediction')
				plt.show()
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


def weather_maps(): 

	float_pos_dict_kona = {'lat':19.556,'lon':-156.538,'datetime':datetime.datetime.now()}
	konaweatherpacioos = ReturnPACIOOSWeather(float_pos_dict_kona)
	konawavespacioos = ReturnPACIOOSWaves(float_pos_dict_kona)
	float_pos_dict_moby = {'lat':20.8,'lon':-157.2,'datetime':datetime.datetime.now()}
	mobyweatherpacioos = ReturnPACIOOSWeather(float_pos_dict_moby)
	mobywavespacioos = ReturnPACIOOSWaves(float_pos_dict_moby)

	for k in range(48):
		XX,YY,ax = HypernavCartopy(konaweatherpacioos,float_pos_dict_kona,konaweatherpacioos.lats,
			konaweatherpacioos.lons,pad=-0.5).get_map()
		q = ax.quiver(XX,YY,konaweatherpacioos.U[k,:,:],konaweatherpacioos.V[k,:,:],scale=510)
		ax.quiverkey(q, X=0.3, Y=1.1, U=5.14444,
				 label='length = 10 kts', labelpos='E')
		plt.title((datetime.datetime.now()+datetime.timedelta(hours=k)).isoformat())
		plt.savefig(file_handler.tmp_file('wind_kona/'+str(k)))
		plt.close()

		XX,YY,ax = HypernavCartopy(konaweatherpacioos,float_pos_dict_kona,konaweatherpacioos.lats,
			konaweatherpacioos.lons,pad=-0.5).get_map()
		plt.pcolormesh(XX,YY,konaweatherpacioos.rain[k,:,:],vmin=0,vmax=0.0005)
		plt.colorbar(label='kilogram meter-2 second-1')
		plt.title((datetime.datetime.now()+datetime.timedelta(hours=k)).isoformat())
		plt.savefig(file_handler.tmp_file('rain_kona/'+str(k)))
		plt.close()

		XX,YY,ax = HypernavCartopy(konawavespacioos,float_pos_dict_kona,konawavespacioos.lats,
			konawavespacioos.lons,pad=-0.5).get_map()
		plt.pcolormesh(XX,YY,konawavespacioos.waves[k,0,:,:],vmin=0,vmax=3)
		plt.colorbar(label='significant waveheight (m)')
		plt.title((datetime.datetime.now()+datetime.timedelta(hours=k)).isoformat())
		plt.savefig(file_handler.tmp_file('waves_kona/'+str(k)))
		plt.close()

		XX,YY,ax = HypernavCartopy(mobyweatherpacioos,float_pos_dict_moby,mobyweatherpacioos.lats,
			mobyweatherpacioos.lons,pad=-0.5).get_map()
		q = ax.quiver(XX,YY,mobyweatherpacioos.U[k,:,:],mobyweatherpacioos.V[k,:,:])
		ax.quiverkey(q, X=0.3, Y=1.1, U=5.14444,
				 label='length = 1o kts', labelpos='E')
		plt.title((datetime.datetime.now()+datetime.timedelta(hours=k)).isoformat())
		plt.savefig(file_handler.tmp_file('wind_moby/'+str(k)))
		plt.close()

		XX,YY,ax = HypernavCartopy(mobyweatherpacioos,float_pos_dict_moby,mobyweatherpacioos.lats,
			mobyweatherpacioos.lons,pad=-0.5).get_map()
		plt.pcolormesh(XX,YY,mobyweatherpacioos.rain[k,:,:],vmin=0,vmax=0.0005)
		plt.colorbar(label='kilogram meter-2 second-1')
		plt.title((datetime.datetime.now()+datetime.timedelta(hours=k)).isoformat())
		plt.savefig(file_handler.tmp_file('rain_moby/'+str(k)))
		plt.close()

		XX,YY,ax = HypernavCartopy(mobywavespacioos,float_pos_dict_moby,mobywavespacioos.lats,
			mobywavespacioos.lons,pad=-0.5).get_map()
		plt.pcolormesh(XX,YY,mobywavespacioos.waves[k,0,:,:],vmin=0,vmax=3)
		plt.colorbar(label='significant waveheight (m)')
		plt.title((datetime.datetime.now()+datetime.timedelta(hours=k)).isoformat())
		plt.savefig(file_handler.tmp_file('waves_moby/'+str(k)))
		plt.close()


	os.chdir(file_handler.tmp_file('wind_kona'))
	os.system('ffmpeg -f image2 -r 2 -i %d.png -c:v libx264 -pix_fmt yuv420p ../wind_kona.mp4')	
	os.chdir(file_handler.tmp_file('rain_kona'))
	os.system('ffmpeg -f image2 -r 2 -i %d.png -c:v libx264 -pix_fmt yuv420p ../rain_kona.mp4')
	os.chdir(file_handler.tmp_file('waves_kona'))
	os.system('ffmpeg -f image2 -r 2 -i %d.png -c:v libx264 -pix_fmt yuv420p ../waves_kona.mp4')
	os.chdir(file_handler.tmp_file('wind_moby'))
	os.system('ffmpeg -f image2 -r 2 -i %d.png -c:v libx264 -pix_fmt yuv420p ../wind_moby.mp4')	
	os.chdir(file_handler.tmp_file('rain_moby'))
	os.system('ffmpeg -f image2 -r 2 -i %d.png -c:v libx264 -pix_fmt yuv420p ../rain_moby.mp4')
	os.chdir(file_handler.tmp_file('waves_moby'))
	os.system('ffmpeg -f image2 -r 2 -i %d.png -c:v libx264 -pix_fmt yuv420p ../waves_moby.mp4')







pword = 'n5vkJ?lw\EmdidlJ'
server_ip = '204.197.4.164'
	
