from GeneralUtilities.Plot.Cartopy.regional_plot import CreteCartopy
from GeneralUtilities.Compute.list import GeoList,VariableList
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots,pdf_particles_compute
from HyperNav.Utilities.Compute.RunParcels import UVPrediction,ParticleDataset,ParticleList,ClearSky
import cartopy.crs as ccrs
import numpy as np
import os
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
from GeneralUtilities.Compute.Depth.depth_utilities import ETopo1Depth
from HyperNav.Utilities.Data.CopernicusMed import CopernicusMed, CreteCopernicus
from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import curl
file_handler = FilePathHandler(ROOT_DIR,'HypernavCreteFutureDeployment')
import cartopy
import cartopy.mpl.geoaxes
from HyperNav.Utilities.Compute.__init__ import ROOT_DIR as COMPUTE_DIR
compute_file_handler = FilePathHandler(COMPUTE_DIR,'RunParcels')

def site_plots(self):
    sites = [(35.8,24.00),(35.8,25.0),(35.5,26.0)]
    lats,lons = zip(*sites)
    self.scatter(lons,lats,350,marker='*',color='Red',zorder=15)

cartopy.mpl.geoaxes.GeoAxesSubplot.site_plots = site_plots


def bathy_plot():
	fig = plt.figure(figsize=(12,12))
	ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
	time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,units = CreteCopernicus.get_dimensions()
	copernicus_instance = CreteCopernicus.load(time[0],time[1])
	XX,YY,ax1 = copernicus_instance.plot(ax=ax1)
	cf = ax1.bathy()
	ax1.site_plots()
	fig.colorbar(cf,ax=[ax1],label='Depth (km)',location='bottom')
	plt.savefig(file_handler.out_file('bathy'))
	plt.close()


def mean_curl_plot():

	def plot_depth_curl(copernicus_instance):
		fig = plt.figure(figsize=(12,12))
		ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
		ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
		cu,cv = copernicus_instance.return_monthly_mean(7,-0)
		cop_curl = CreteCopernicus.calculate_curl(cu,cv)


		plot_max = np.nanmax(cop_curl)
		plot_min = np.nanmin(cop_curl)

		XX,YY,ax1 = copernicus_instance.plot(ax=ax1)
		ax1.pcolor(XX,YY,cop_curl,vmin = plot_min, vmax = plot_max,cmap='PiYG')
		ax1.quiver(XX[::3,::3],YY[::3,::3],cu[::3,::3],cv[::3,::3],scale=2)
		ax1.site_plots()
		ax1.title.set_text('Surface')

		cu,cv = copernicus_instance.return_monthly_mean(7,-700)
		cop_curl = CreteCopernicus.calculate_curl(cu,cv)
		plot_max = np.nanmax(cop_curl)
		plot_min = np.nanmin(cop_curl)
		XX,YY,ax2 = copernicus_instance.plot(ax=ax2)
		ax2.pcolor(XX,YY,cop_curl,vmin = plot_min, vmax = plot_max,cmap='PiYG')
		ax2.site_plots()
		q = ax2.quiver(XX[::3,::3],YY[::3,::3],cu[::3,::3],cv[::3,::3],scale=2)
		ax1.quiverkey(q,X=-0.3, Y=1.1, U=1,
             label='Quiver key, length = 1 m/s', labelpos='E')
		ax2.title.set_text('700 meters')
		PCM = ax2.get_children()[0]
		fig.colorbar(PCM,ax=[ax1,ax2],label='Curl ($s^{-1}$)',location='bottom')
		plt.savefig(file_handler.out_file('curl_comparison'))
		plt.close()



	copernicus_instance = CreteCopernicus.load(datetime.datetime(2021,7,1),datetime.datetime(2021,7,28))
	plot_depth_curl(copernicus_instance)

	def plot_depth_quiver(copernicus_instance,depth,level):
		fig = plt.figure(figsize=(12,12))
		ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		cu,cv = copernicus_instance.return_monthly_mean(7,CreteCopernicus.max_depth)
		u_std,v_std = copernicus_instance.return_monthly_std(7,CreteCopernicus.max_depth)

		XX,YY,ax1 = copernicus_instance.plot(ax=ax1)
		q = ax1.quiver(XX[::2],YY[::2],cu[::2],cv[::2],scale=level,zorder=10)
		ax1.quiverkey(q,X=0.5, Y=1.1, U=level/5,
             label='Quiver key, length = '+str(level/5)+' m/s', labelpos='E')
		ax1.pcolor(XX,YY,u_std**2+v_std**2)
		PCM = ax1.get_children()[1]
		fig.colorbar(PCM,ax=[ax1],label='Velocity Std ($m^2s^{-2}$)',location='bottom')
		ax1.title.set_text('Copernicus')
		plt.savefig(file_handler.out_file('mean'+str(CreteCopernicus.max_depth)))
		plt.close()

	plot_depth_quiver(copernicus_instance,0,10) 	
	plot_depth_quiver(copernicus_instance,700,2.)

	def curl(copernicus_instance,depth):
		fig = plt.figure(figsize=(12,12))
		ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		cu,cv = copernicus_instance.return_monthly_mean(7,CreteCopernicus.max_depth)
		cop_curl = CreteCopernicus.calculate_curl(cu,cv)
		XX,YY,ax1 = copernicus_instance.plot(ax=ax1)
		pc = ax1.pcolor(XX,YY,cop_curl,cmap='PiYG')
		ax1.site_plots()
		fig.colorbar(pc,ax=[ax1],label='Curl ($s^{-1}$)',location='bottom')
		plt.savefig(file_handler.out_file('curl_'+str(CreteCopernicus.max_depth)))
		plt.close()
	curl(copernicus_instance,0)
	curl(copernicus_instance,700)

def ts_plot():
	lat = 35.8
	lon = 25.0
	start_date = datetime.datetime(2021,7,1)
	end_date = datetime.datetime(2021,7,31)
	fig,fig1 = CreteCopernicus.get_sal_temp_profiles(lat,lon,start_date,end_date)
	fig.savefig(file_handler.out_file('site_2_ts'))
	fig1.savefig(file_handler.out_file('site_2_density'))

def crete_shear_movie():
	uv_class = 	CreteCopernicus.load(datetime.datetime(2021,7,1),datetime.datetime(2021,8,1))
	lat = 35.8
	lon = 25.0
	mask = [(x>datetime.datetime(2021,7,1))&(x<datetime.datetime(2021,8,1)) for x in uv_class.time]
	shear_movie(uv_class,mask,file_handler,lat,lon)


def CreteParticlesCompute():
	date_start = datetime.datetime(2021,7,11)
	date_end = datetime.datetime(2021,7,25)
	uv_class = CreteCopernicus.load(date_start-datetime.timedelta(days=3),date_end+datetime.timedelta(days=3))

	start_time = date_start.timestamp()
	end_time = date_end.timestamp()
	uv_class.depth[0]=0
	uv_class = uv_class.subsample_depth(4,max_depth=-650)
	uv_class = uv_class.subsample_time_u_v(3)
	data,dimensions = uv_class.return_parcels_uv(date_start-datetime.timedelta(days=2),date_end+datetime.timedelta(days=2))
	lat = 35.74
	lon = 25.07
	surface_time = 5400
	vertical_speed = 0.076
	from GeneralUtilities.Compute.list import TimeList
	for depth in [300,400,500,600]:
		argo_cfg = {'lat': lat, 'lon': lon, 'target_lat': np.nan, 'target_lon': np.nan,
					'time': start_time, 'end_time': end_time, 'depth': 10, 'min_depth': 10, 'drift_depth': abs(depth),
					'max_depth': abs(depth),
					'surface_time': surface_time, 'total_cycle_time': 24*3600,
					'vertical_speed': vertical_speed,
					}

		dist_loc = []
		filename = 'site_2_'+str(depth)
		if not os.path.isfile(compute_file_handler.tmp_file(filename+'.nc')):
			prediction = UVPrediction(argo_cfg,data,dimensions)
			prediction.create_prediction()
			os.rename(compute_file_handler.tmp_file('Uniform_out.nc'),compute_file_handler.tmp_file(filename+'.nc'))
		TimeList.set_ref_date(date_start)
		nc = ParticleDataset(compute_file_handler.tmp_file(filename+'.nc'))
		for delta in [datetime.timedelta(days=x,seconds=0) for x in range(14)]:
			lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
			dist_loc.append((lat_center,lon_center))
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		XX,YY,ax1 = uv_class.plot(ax=ax1)
		for n in range(nc['lat'].shape[0]):
			ax1.scatter(nc['lon'][n,:],nc['lat'][n,:],s=0.2)
			ax1.scatter(nc['lon'][n,-1],nc['lat'][n,-1],s=1,c='k',marker='x',zorder=20)
		plt.title(filename)
		plt.savefig(file_handler.out_file(filename))
		plt.close()		

		fig = plt.figure()
		data_list = []

		for n in range(nc['lat'].shape[0]):
			data_list.append(len(nc['z'][:][n,:][~nc['z'][:][n,:].mask]))
		plt.hist(data_list,bins=100)
		plt.xlabel('Time Step')
		plt.title('Timestep Ran Aground')
		plt.savefig(file_handler.out_file(filename+'_hist'))
		plt.close()		
		dist_lat,dist_lon = zip(*dist_loc)
		geolist = GeoList([geopy.Point(x) for x in dist_loc])
		EEZ_list = []
		for k,dummy in enumerate(geolist.to_shapely()):
			print(k)
			EEZ_list += self.df[self.df.contains(dummy)].TERRITORY1.tolist()	# only choose coordinates within the ocean basin of interest


		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		XX,YY,ax1 = uv_class.plot(ax=ax1)
		ax1.scatter(dist_lon,dist_lat)
		# lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
		# ax1.scatter(lon_center,lat_center)
		ax1.scatter(float_pos_dict['lon'],float_pos_dict['lat'],marker='x',c='k',linewidth=6,s=250,)
		plt.savefig(file_handler.out_file(filename))
		plt.close()