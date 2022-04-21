from GeneralUtilities.Plot.Cartopy.regional_plot import KonaCartopy
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots,pdf_particles_compute
from HyperNav.Utilities.Compute.RunParcels import UVPrediction,ParticleDataset,ParticleList,ClearSky
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
import cartopy.crs as ccrs
import numpy as np
import os
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
from GeneralUtilities.Data.depth.depth_utilities import ETopo1Depth


file_handler = FilePathHandler(ROOT_DIR,'HypernavHawaiiFutureDeployment')


class FutureHawaiiCartopy(RegionalBase):
    llcrnrlon=-159
    llcrnrlat=16
    urcrnrlon=-154
    urcrnrlat=22
    def __init__(self,*args,**kwargs):
        print('I am plotting Kona')
        super().__init__(*args,**kwargs)

class HYCOMFutureHawaii(HYCOMHawaii):
	PlotClass = FutureHawaiiCartopy
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)


def hawaii_mean_monthly_plot():
	uv_class = HYCOMFutureHawaii.load()
	mean_monthly_plot(uv_class,file_handler,month=12)


def hawaii_quiver_movie():
	uv_class = HYCOMFutureHawaii.load()
	mask = [(x>datetime.datetime(2020,12,1))&(x<datetime.datetime(2020,12,30)) for x in uv_class.time]
	quiver_movie(uv_class,mask,file_handler)


def hawaii_shear_movie():
	uv_class =  HYCOMFutureHawaii.load()
	lat = 20
	lon = -156.1
	mask = [(x>datetime.datetime(2020,12,1))&(x<datetime.datetime(2020,12,30)) for x in uv_class.time]
	shear_movie(uv_class,mask,file_handler,lat,lon)

def hawaii_eke():
	uv_class =  HYCOMFutureHawaii.load()
	eke_plots(uv_class,file_handler)

def hawaii_particles_compute():
	uv_class = HYCOMFutureHawaii
	float_list = [({'lat':19.5,'lon':-156.3,'time':datetime.datetime(2015,11,20)},'site_1')]
	pdf_particles_compute(uv_class,float_list,file_handler)

def future_prediction():
	date_start = datetime.datetime(2021,4,15)
	date_end = datetime.datetime(2021,5,15)
	uv_class = HYCOMFutureHawaii.load(date_start-datetime.timedelta(days=1),date_end)
	data,dimensions = uv_class.return_parcels_uv(date_start,date_end)
	lats = np.arange(19.4,19.75,0.1)
	lons = np.arange(-156.7,-156.2,0.1)
	pl = ParticleList()

	X,Y = np.meshgrid(lons,lats)

	lons = X.flatten()
	lats = Y.flatten()

	for lat,lon in zip(lats,lons):
		time = [date_start]
		start_time = date_start.timestamp()
		end_time = date_end.timestamp()
		drift_depth = -700
		surface_time = 5400
		vertical_speed = 0.076
		total_cycle_time = (date_start - date_end).seconds
		argo_cfg = {'lat': lat, 'lon': lon, 'target_lat': np.nan, 'target_lon': np.nan,
					'time': start_time, 'end_time': end_time, 'depth': 10, 'min_depth': 10, 'drift_depth': abs(drift_depth),
					'max_depth': abs(700),
					'surface_time': surface_time, 'total_cycle_time': total_cycle_time,
					'vertical_speed': vertical_speed,
					}

		data,dimensions = uv_class.return_parcels_uv(date_start-datetime.timedelta(hours=1),date_end+datetime.timedelta(days=3))
		prediction = UVPrediction(argo_cfg,data,dimensions)
		prediction.create_prediction()
		nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
		pl.append(nc)
	plt.rcParams["figure.figsize"] = (15,15)

	from matplotlib.colors import LinearSegmentedColormap
	KonaCartopy.llcrnrlon=-157.8
	KonaCartopy.llcrnrlat=18.5
	KonaCartopy.urcrnrlon=-155.8
	KonaCartopy.urcrnrlat=20.5

	
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
	lat_list = []
	lon_list = []
	for r,timedelta in enumerate([datetime.timedelta(hours=x) for x in range(24*28)[::12]]):
		scatter_list = [x.get_cloud_center(timedelta) for x in pl]
		lat,lon,lat_std,lon_std = zip(*scatter_list)
		lat_list.append(list(lat))
		lon_list.append(list(lon))
		DUM,DUM,ax = KonaCartopy().get_map()
		ax.scatter(lon,lat,marker='X',zorder=15)
		# ax.scatter(lon[34],lat[34],c='r',marker='X',zorder=16)

		lat_holder = np.vstack(lat_list)
		lon_holder = np.vstack(lon_list)
		for k in range(lat_holder.shape[1]):
			ax.plot(lon_holder[:,k],lat_holder[:,k],'b',alpha=0.2)
		# ax.plot(lon_holder[:,34],lat_holder[:,34],'r',zorder=16)

		ax.contourf(XX,YY,plot_data,levels,cmap=bathy,animated=True,vmax=6,vmin=0)
		plt.title(date_start+timedelta)
		plt.savefig(file_handler.out_file('deployment_movie/'+str(r)))
		plt.close()
	os.chdir(file_handler.out_file('deployment_movie'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")


def clear_days_prediction():
	uv_class = HYCOMFutureHawaii.load(datetime.datetime(2015,11,15),datetime.datetime(2016,3,15))
	float_list = [({'lat':19.5,'lon':-156.4,'time':datetime.datetime(2015,11,15)},'site_1')]
	dict_list = []
	for month,filename in [(12,'Dec'),(1,'Jan')]:
		dict_list.append((month,ClearSky(filename)))
	clear_sky_dict = dict(dict_list)
	pl = ParticleList()
	for float_pos_dict,filename in float_list:
		uv_class.time.set_ref_date(float_pos_dict['time'])
		for start_day in [5]*3:
			float_pos_dict['time'] = float_pos_dict['time']+datetime.timedelta(days=start_day)
			data,dimensions = uv_class.return_parcels_uv(float_pos_dict['time'],days_delta=61)
			prediction = UVPrediction(float_pos_dict,data,dimensions)
			prediction.create_prediction(ArgoVerticalMovement600,days=60.)
			nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
			pl.append(nc)
		for k,timedelta in enumerate([datetime.timedelta(days=x) for x in range(59)]):
			XX,YY,ax = uv_class.plot()
			pl.plot_density(timedelta,[uv_class.lons,uv_class.lats],ax)
			plt.savefig(file_handler.out_file('pdf_movie_'+filename+'/'+str(k)))
			plt.close()
		os.chdir(file_handler.out_file('pdf_movie_'+filename+'/'))
		os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")
		clear_sky_plot = []
		aot_plot = []
		aot_percent_plot = []
		for time_delta in [datetime.timedelta(days = x ) for x in np.arange(2,60,2).tolist()]:
			lats,lons = pl.get_cloud_snapshot(time_delta)
			time = pl.get_time(time_delta)
			clear_sky_holder = [clear_sky_dict[12].return_clear_sky(y,x) for x,y,t in zip(lons,lats,time)]
			clear_sky_plot.append((time_delta.days,np.nanmean(clear_sky_holder)))
			aot_holder = [clear_sky_dict[12].return_aot(y,x) for x,y,t in zip(lons,lats,time)]
			aot_plot.append((time_delta.days,np.nanmean(aot_holder)))
			aot_percent_holder = [clear_sky_dict[12].return_aot_percent(y,x) for x,y,t in zip(lons,lats,time)]
			aot_percent_plot.append((time_delta.days,np.nanmean(aot_percent_holder)))			
		days,sky = zip(*clear_sky_plot)
		Y = np.cumsum(np.array(sky)*.02)
		plt.plot(days,Y)
		plt.subplot(2,1,1)
		plt.xlabel('Days')
		plt.ylabel('Cumulative Chance of Clear Sky (%)')
		plt.plot(days,Y)
		plt.subplot(2,1,2)
		plt.plot(days,sky)
		plt.xlabel('Days')
		plt.ylabel('Chance of Clear Sky (%)')
		plt.savefig('clear_sky')
		plt.close()
		days,sky = zip(*aot_percent_plot)
		Y = np.cumsum(np.array(sky)*.02)
		plt.plot(days,Y*100)
		plt.subplot(2,1,1)
		plt.xlabel('Days')
		plt.ylabel('Cumulative Chance of Low AOT (%)')
		plt.plot(days,Y)
		plt.subplot(2,1,2)
		plt.plot(days,sky)
		plt.xlabel('Days')
		plt.ylabel('Chance of Low AOT (%)')
		plt.savefig('aot')
		plt.close()
		plt.close()




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
	