from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots,pdf_particles_compute
from HyperNav.Utilities.Compute.RunParcels import create_prediction,ParticleDataset,ParticleList
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
import cartopy.crs as ccrs
import numpy as np
import os
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
from HyperNav.Utilities.Data.CopernicusGlobal import HawaiiCopernicus
file_handler = FilePathHandler(ROOT_DIR,'HawaiiCopernicus')


class HawaiiCartopy(RegionalBase):
	lat = 17.402
	lon = -155.805

	llcrnrlon=lon-0.3
	llcrnrlat=lat-0.3
	urcrnrlon=lon+0.3
	urcrnrlat=lat+0.3

	def __init__(self,*args,**kwargs):
		print('I am plotting Hawaii')
		super().__init__(*args,**kwargs)



def hawaii_particles_compute():
	date_start = datetime.datetime.today()
	date_end = datetime.datetime(2024,12,19)
	uv_class = HawaiiCopernicus.load(date_start,date_end)
	start_time = date_start.timestamp()
	end_time = date_end.timestamp()
	uv_class.depths[0]=0
	# uv_class = uv_class.subsample_depth(4,max_depth=-650)
	# uv_class = uv_class.subsample_time_u_v(3)
	data,dimensions = uv_class.return_parcels_uv(date_start-datetime.timedelta(hours=48),date_end+datetime.timedelta(hours=48))
	lat = 17.402
	lon = -155.805
	assert (lat>min(dimensions['lat']))&(lat<max(dimensions['lat']))
	assert (lon>min(dimensions['lon']))&(lon<max(dimensions['lon']))
	assert (start_time>min(dimensions['time']))&(start_time<max(dimensions['time']))
	assert (end_time>min(dimensions['time']))&(end_time<max(dimensions['time']))

	surface_time = 300
	vertical_speed = 0.076
	for depth in [100,200,300,400,425,500,600,700]:
		argo_cfg = {'lat': lat, 'lon': lon, 'target_lat': np.nan, 'target_lon': np.nan,
					'time': start_time, 'end_time': end_time, 'depth': 10, 'min_depth': 10, 'drift_depth': abs(depth),
					'max_depth': abs(depth),
					'surface_time': surface_time, 'total_cycle_time': 8*3600,
					'vertical_speed': vertical_speed,
					}
		create_prediction(argo_cfg,data,dimensions,'temp'+str(depth)+'.zarr',n_particles=500)		
	loc_dict = {}
	for depth in [100,200,300,400,425,500,600,700]:
		nc = ParticleDataset('temp'+str(depth)+'.zarr')
		dist_loc = []
		for delta in [datetime.timedelta(days=0,seconds=28800*x) for x in range(4)]:
			lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
			dist_loc.append((lat_center,lon_center))
		loc_dict[depth]=dist_loc
	uv_class.PlotClass = HawaiiCartopy
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = uv_class.plot(ax=ax)
	for depth in [100,200,300,400,425,500,600,700]:
		lats,lons = zip(*loc_dict[depth])
		ax1.scatter(lons,lats,label=str(depth)+' m')
		ax1.scatter(lons[-1],lats[-1],marker='*',color='k')
		ax1.plot(lons,lats)
	ax1.legend()