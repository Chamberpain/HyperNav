from GeneralUtilities.Plot.Cartopy.regional_plot import KonaCartopy
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots,pdf_particles_compute
from HyperNav.Utilities.Compute.RunParcels import UVPrediction,ParticleDataset,ParticleList,ClearSky
import cartopy.crs as ccrs
import numpy as np
import os
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50
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

def add_zero_boundary_conditions(_matrix):
		_matrix[:,:,:,0] = 0
		_matrix[:,:,:,-1] = 0
		_matrix[:,:,0,:] = 0
		_matrix[:,:,-1,:] = 0
		return _matrix

class HYCOMFutureHawaii(HYCOMHawaii):
	PlotClass = FutureHawaiiCartopy
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.u = add_zero_boundary_conditions(self.u)
		self.v = add_zero_boundary_conditions(self.v)		

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
	date_start = datetime.datetime(2021,11,14)
	date_end = datetime.datetime(2021,11,21)
	HYCOMFutureHawaii.load(date_start-datetime.timedelta(days=1),date_end)
	uv_class = HYCOMFutureHawaii.load(date_start-datetime.timedelta(days=1),date_end)
	lats = np.arange(19.4,19.65,0.05)
	lons = np.arange(-156.5,-156.2,0.05)
	X,Y = np.meshgrid(lons,lats)
	lons = X.flatten()
	lats = Y.flatten()
	dates = [date_start]*len(lons)
	keys = ['lat','lon','time']
	float_list = [dict(zip(keys,list(x))) for x in zip(lats,lons,dates)]
	pl = ParticleList()
	for float_pos_dict in float_list:
		uv_class.time.set_ref_date(float_pos_dict['time'])
		data,dimensions = uv_class.return_parcels_uv(float_pos_dict['time']-datetime.timedelta(hours=1),days_delta=7)
		prediction = UVPrediction(float_pos_dict,data,dimensions)
		prediction.create_prediction(ArgoVerticalMovement600,days=6)
		nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
		pl.append(nc)
	plt.rcParams["figure.figsize"] = (15,15)

	from matplotlib.colors import LinearSegmentedColormap
	KonaCartopy.llcrnrlon=-156.6
	KonaCartopy.llcrnrlat=19.2
	KonaCartopy.urcrnrlon=-156
	KonaCartopy.urcrnrlat=19.8

	
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
	for r,timedelta in enumerate([datetime.timedelta(hours=x) for x in range(24*6)]):
		scatter_list = [x.get_cloud_center(timedelta) for x in pl]
		lat,lon,lat_std,lon_std = zip(*scatter_list)
		lat_list.append(list(lat))
		lon_list.append(list(lon))
		DUM,DUM,ax = KonaCartopy().get_map()
		ax.scatter(lon,lat,marker='X',zorder=15)
		ax.scatter(lon[16],lat[16],c='r',marker='X',zorder=16)

		lat_holder = np.vstack(lat_list)
		lon_holder = np.vstack(lon_list)
		for k in range(lat_holder.shape[1]):
			ax.plot(lon_holder[:,k],lat_holder[:,k],'b',alpha=0.2)
		ax.plot(lon_holder[:,16],lat_holder[:,16],'r',zorder=16)

		ax.contourf(XX,YY,plot_data,levels,cmap=bathy,animated=True,vmax=6,vmin=0)
		plt.title(date_start+timedelta)
		plt.savefig(file_handler.out_file('deployment_movie/'+str(r)))
		plt.close()
	os.chdir(file_handler.out_file('deployment_movie'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")


def status_update():
	date_start = datetime.datetime(2021,11,16,21)
	date_end = datetime.datetime(2021,11,25)
	HYCOMFutureHawaii.load(date_start-datetime.timedelta(days=1),date_end)
	uv_class = HYCOMFutureHawaii.load(date_start-datetime.timedelta(days=1),date_end)
	lats = np.arange(19.4,19.65,0.05)
	lons = np.arange(-156.5,-156.2,0.05)
	X,Y = np.meshgrid(lons,lats)
	lons = X.flatten()
	lats = Y.flatten()
	dates = [date_start]*len(lons)
	keys = ['lat','lon','time']
	float_list = [dict(zip(keys,[19.6,-156.7,date_start]))]
	pl = ParticleList()
	for float_pos_dict in float_list:
		uv_class.time.set_ref_date(float_pos_dict['time'])
		data,dimensions = uv_class.return_parcels_uv(float_pos_dict['time']-datetime.timedelta(hours=1),days_delta=7)
		prediction = UVPrediction(float_pos_dict,data,dimensions)
		prediction.create_prediction(ArgoVerticalMovement700,days=6)
		nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
		pl.append(nc)


	float_track = [(geopy.Point(19.5998,-156.6906),datetime.datetime(2021,11,16,21,00,00)),
	(geopy.Point(19.6091,-156.6516),datetime.datetime(2021,11,17,2,41,0)),
	(geopy.Point(19.6663,-156.6161),datetime.datetime(2021,11,17,18,13,10)),
	(geopy.Point(19.7465,-156.5401),datetime.datetime(2021,11,18,11,54,0)),
	(geopy.Point(19.7959,-156.5244),datetime.datetime(2021,11,19,3,14,00)),
	(geopy.Point(19.8534,-156.5333),datetime.datetime(2021,11,19,22,37,40))]
	dist_list = []
	point_list = []
	for float_pos,time in float_track:
		time_diff = time-date_start
		lat,lon,DUM,DUM = nc.get_cloud_center(time_diff)
		predict_point = geopy.Point(lat,lon)
		dist_list.append((time_diff.total_seconds()/86400.,geopy.distance.GreatCircleDistance(float_pos,predict_point).nm))
		point_list.append((float_pos,predict_point)) 
	x,y = zip(*dist_list)
	plt.plot(x,y)
	plt.xlabel('Time (days)')
	plt.ylabel('Error (nm)')
	plt.savefig(file_handler.out_file('53_error '))
	plt.close()


	plt.rcParams["figure.figsize"] = (15,15)

	from matplotlib.colors import LinearSegmentedColormap
	KonaCartopy.llcrnrlon=-156.8
	KonaCartopy.llcrnrlat=19.2
	KonaCartopy.urcrnrlon=-156
	KonaCartopy.urcrnrlat=20.3

	
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
	predict_lon = [x[1].longitude for x in point_list]
	predict_lat = [x[1].latitude for x in point_list]
	actual_lon = [x[0].longitude for x in point_list]
	actual_lat = [x[0].latitude for x in point_list]

	DUM,DUM,ax = KonaCartopy().get_map()
	ax.plot(predict_lon,predict_lat,label='Prediction')
	ax.plot(actual_lon,actual_lat,label='Actual')
	ax.contourf(XX,YY,plot_data,levels,cmap=bathy,animated=True,vmax=6,vmin=0)
	plt.legend(loc='lower left')
	plt.savefig(file_handler.out_file('53_track'))
	plt.close()


	date_start = float_track[-1][1]
	date_end = datetime.datetime(2021,11,28)
	HYCOMFutureHawaii.load(date_start-datetime.timedelta(days=1),date_end)
	uv_class = HYCOMFutureHawaii.load(date_start-datetime.timedelta(days=1),date_end)
	lats = np.arange(19.4,19.65,0.05)
	lons = np.arange(-156.5,-156.2,0.05)
	X,Y = np.meshgrid(lons,lats)
	argo_behavior_list = [ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,
	ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50]
	keys = ['lat','lon','time']
	float_pos_dict = dict(zip(keys,[float_track[-1][0].latitude,float_track[-1][0].longitude,date_start]))
	pl = ParticleList()
	for argo_behavior in argo_behavior_list:
		uv_class.time.set_ref_date(float_pos_dict['time'])
		data,dimensions = uv_class.return_parcels_uv(float_pos_dict['time']-datetime.timedelta(hours=1),days_delta=7)
		prediction = UVPrediction(float_pos_dict,data,dimensions)
		prediction.create_prediction(argo_behavior,days=3)
		nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
		pl.append(nc)

	DUM,DUM,ax = KonaCartopy().get_map()
	for particle,name in zip(pl,['700 m','600 m','500 m','400 m','300 m','200 m','100 m','50 m']):
		point_list = [particle.get_cloud_center(datetime.timedelta(hours=int(x))) for x in np.arange(0,120,6)]
		lats,lons, DUM,DUM = zip(*point_list)
		ax.plot(lons,lats,label=name)
	plt.legend()
	plt.savefig(file_handler.out_file('53_prediction'))


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