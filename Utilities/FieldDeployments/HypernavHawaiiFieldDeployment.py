from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots
import cartopy.crs as ccrs
import numpy as np
import os
file_handler = FilePathHandler(ROOT_DIR,'HypernavHawaiiFieldDeployment')


def hawaii_mean_monthly_plot():
	uv_class = HYCOMHawaii.load()
	mean_monthly_plot(uv_class,file_handler,month=6)


def hawaii_quiver_movie():
	uv_class = HYCOMHawaii.load()
	mask = [(x>datetime.datetime(2021,6,6))&(x<datetime.datetime(2021,6,24)) for x in uv_class.time]
	quiver_movie(uv_class,mask,file_handler)


def hawaii_shear_movie():
	uv_class =  HYCOMHawaii.load()
	lat = 19.5
	lon = -156.3
	mask = [(x>datetime.datetime(2021,6,6))&(x<datetime.datetime(2021,6,24)) for x in holder.time]
	shear_movie(uv_class,mask,file_handler,lat,lon)

def hawaii_eke():
	uv_class =  HYCOMHawaii.load()
	eke_plots(uv_class,file_handler)


def HawaiiParticlesCompute():
	from HyperNav.Utilities.Data.Instrument import HI21A55, HI21A54, HI21B55, HI21B54
	from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
	from GeneralUtilities.Compute.constants import degree_dist, seconds_per_day
	import math
	import cartopy.crs as ccrs
	import os

	uv = HYCOMHawaii.load()
	dist_list = []
	baro_list = []
	try:
		del delta_x
		del delta_y
	except NameError:
		pass
	for Hypernav in [HI21A55(), HI21A54(), HI21B55(), HI21B54()]:
		for time_idx in range(len(Hypernav.time)-1):
			date = Hypernav.time[time_idx]
			float_pos_dict = Hypernav.return_float_pos_dict(date)
			baro_loc = []
			try:
				baro,dimensions = uv.return_parcels_uv(date)
				baro['U'] = baro['U']-delta_x
				baro['V'] = baro['V']-delta_y
				prediction = UVPrediction(float_pos_dict,baro,dimensions)
				prediction.create_prediction(ArgoVerticalMovement600)
				nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
				for delta in [datetime.timedelta(days=x) for x in [1,2,3]]:
					projected_date = date+delta
					if projected_date>max(Hypernav.time):
						continue 
					lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
					baro_loc.append((lat_center,lon_center))
					surface_loc = geopy.Point(Hypernav.lats[time_idx+delta.days],Hypernav.lons[time_idx+delta.days])
					baro_list.append((geopy.distance.great_circle(geopy.Point(lat_center,lon_center),surface_loc).nm,delta.days))	
			except NameError:
				pass

			data,dimensions = uv.return_parcels_uv(date)
			prediction = UVPrediction(float_pos_dict,data,dimensions)
			prediction.create_prediction(ArgoVerticalMovement600)
			nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
			dist_loc = []
			for delta in [datetime.timedelta(days=x) for x in [1,2,3]]:
				projected_date = date+delta
				if projected_date>max(Hypernav.time):
					continue 
				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
				dist_loc.append((lat_center,lon_center))
				surface_loc = geopy.Point(Hypernav.lats[time_idx+delta.days],Hypernav.lons[time_idx+delta.days])
				if delta.days==1:
					delta_y = (lat_center-Hypernav.lats[time_idx+delta.days])*degree_dist*1000/seconds_per_day
					delta_x = (lon_center-Hypernav.lons[time_idx+delta.days])*math.cos(math.radians(lat_center))*degree_dist*1000/seconds_per_day
				dist_list.append((geopy.distance.great_circle(geopy.Point(lat_center,lon_center),surface_loc).nm,delta.days))
			fig = plt.figure()
			ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
			XX,YY,ax1 = uv.plot(ax=ax1)
			if baro_loc:
				baro_lat,baro_lon = zip(*baro_loc)
			else:
				baro_lat = []
				baro_lon = []
			if dist_loc:
				dist_lat,dist_lon = zip(*dist_loc)
			else:
				dist_lat = []
				dist_lon = []
			ax1.scatter(baro_lon,baro_lat,alpha=0.3,label='Adjusted Model')
			ax1.plot(baro_lon,baro_lat,alpha=0.3)
			ax1.scatter(dist_lon,dist_lat,alpha=0.3,label='Model')
			ax1.plot(dist_lon,dist_lat,alpha=0.3)
			ax1.scatter(Hypernav.lons[time_idx:(time_idx+3)],Hypernav.lats[time_idx:(time_idx+3)],alpha=0.3,label='Real')
			ax1.plot(Hypernav.lons[time_idx:(time_idx+3)],Hypernav.lats[time_idx:(time_idx+3)],alpha=0.3)
			ax1.legend(loc=2)
			plt.title(Hypernav.label+' '+date.ctime())
			plt.savefig(file_handler.out_file(Hypernav.label+'/'+str(time_idx)))
			plt.close()
		os.chdir(file_handler.out_file(Hypernav.label+'/'))
		os.system("ffmpeg -r 1 -i %01d.png -vcodec mpeg4 -y movie.mp4")


	dist_error,dist_days = zip(*dist_list)
	baro_error,baro_days = zip(*baro_list)
	dist_mean = []
	dist_std = []
	baro_mean = []
	baro_std = []
	days = []
	for day in np.unique(dist_days):
		days.append(day)
		dist_mean.append(np.array(dist_error)[np.array(dist_days)==day].mean())
		baro_mean.append(np.array(baro_error)[np.array(baro_days)==day].mean())
		dist_std.append(np.array(dist_error)[np.array(dist_days)==day].std())
		baro_std.append(np.array(baro_error)[np.array(baro_days)==day].std())
	plt.subplot(2,1,1)
	plt.scatter(days,dist_mean,c='orange',label='Base')
	plt.errorbar(days,dist_mean,dist_std,ecolor='orange',alpha=0.3)
	plt.legend()
	plt.subplot(2,1,2)
	plt.scatter(days,baro_mean,c='blue',label='Adjusted')
	plt.errorbar(days,baro_mean,baro_std,ecolor='blue',alpha=0.3)
	plt.legend()
	plt.savefig(file_handler.out_file('HawaiiStats'))
	plt.close()
