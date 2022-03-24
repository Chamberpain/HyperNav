import datetime
import numpy as np
from GeneralUtilities.Compute.list import find_nearest, TimeList
from parcels import DiffusionUniformKh, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4,AdvectionDiffusionM1, plotTrajectoriesFile
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50
from HyperNav.Utilities.Data.API import SiteAPI
from GeneralUtilities.Data.depth.depth_utilities import PACIOOS as Depth
from GeneralUtilities.Plot.Cartopy.eulerian_plot import HypernavCartopy
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import os
import requests
from pydap.client import open_url
from GeneralUtilities.Plot.Cartopy.regional_plot import KonaCartopy

vert_move_dict={50:ArgoVerticalMovement50,100:ArgoVerticalMovement100,200:ArgoVerticalMovement200,
				300:ArgoVerticalMovement300,400:ArgoVerticalMovement400,500:ArgoVerticalMovement500,
				600:ArgoVerticalMovement600,700:ArgoVerticalMovement700}




file_handler = FilePathHandler(ROOT_DIR,'PACIOOS')

class PACIOOSDataOpenAndParse(object):
	base_html = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap/'
	def __init__(self,float_pos_dict):
		plot_pad = 3
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
		self.time = TimeList(time_in_datetime)
		self.time.set_ref_date(time_since)

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


def special_happy_navigation_time():
	float_pos_dict = {'lat':19.9313,'lon':-156.5501,'datetime':datetime.datetime(2021,11,20,22,45)}
	uv = ReturnPACIOOSUV(float_pos_dict)
	pl = ParticleList()
	prediction = UVPrediction(float_pos_dict,uv.data,uv.dimensions)
	for float_behavior in [AltArgoVerticalMovement25,AltArgoVerticalMovement50,AltArgoVerticalMovement75,AltArgoVerticalMovement100,AltArgoVerticalMovement125]:
		prediction.create_prediction(float_behavior,days=2)
		nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
		pl.append(nc)
	from matplotlib.colors import LinearSegmentedColormap
	KonaCartopy.llcrnrlon=float_pos_dict['lon']-.5
	KonaCartopy.llcrnrlat=float_pos_dict['lat']-.5
	KonaCartopy.urcrnrlon=float_pos_dict['lon']+.5
	KonaCartopy.urcrnrlat=float_pos_dict['lat']+.5


	
	r_start = 0.0
	g_start = 0.5
	b_start = 0.5
	delta = 0.08
	cdict = {'red':  ((r_start, g_start, b_start),
			(r_start+0.2, g_start+delta, b_start+delta),
			(r_start+0.4, g_start+2*delta, b_start+2*delta),
			(r_start+0.6, g_start+3*delta, b_start+3*delta),
			(r_start+0.8, g_start+4*delta, b_start+4*delta),
			(r_start+1.0, g_start+5*delta, b_start+5*delta)),

	 'green':((r_start, g_start, b_start),
			(r_start+0.2, g_start+delta, b_start+delta),
			(r_start+0.4, g_start+2*delta, b_start+2*delta),
			(r_start+0.6, g_start+3*delta, b_start+3*delta),
			(r_start+0.8, g_start+4*delta, b_start+4*delta),
			(r_start+1.0, g_start+5*delta, b_start+5*delta)),

	 'blue': ((r_start, g_start, b_start),
			(r_start+0.2, g_start+delta, b_start+delta),
			(r_start+0.4, g_start+2*delta, b_start+2*delta),
			(r_start+0.6, g_start+3*delta, b_start+3*delta),
			(r_start+0.8, g_start+4*delta, b_start+4*delta),
			(r_start+1.0, g_start+5*delta, b_start+5*delta)),
	}
	bathy = LinearSegmentedColormap('bathy', cdict)	

	depth = ETopo1Depth.load().regional_subsample(KonaCartopy.urcrnrlon,KonaCartopy.llcrnrlon,KonaCartopy.urcrnrlat,KonaCartopy.llcrnrlat)
	plot_data = -depth.z/1000.
	XX,YY = np.meshgrid(depth.lon,depth.lat)
	levels = [0,1,2,3,4,5,6]
	DUM,DUM,ax = KonaCartopy().get_map()
	ax.contourf(XX,YY,plot_data,levels,cmap=bathy,animated=True,vmax=6,vmin=0)

	for particle,name in zip(pl,['25 m','50 m','75 m','100 m','125 m','200m']):
		point_list = [particle.get_cloud_center(datetime.timedelta(hours=int(x))) for x in np.arange(0,120,6)]
		lats,lons, DUM,DUM = zip(*point_list)
		ax.plot(lons,lats,label=name)
	plt.legend(loc='lower left')
	plt.savefig('happy_funtime_argo_navigation')

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
	
