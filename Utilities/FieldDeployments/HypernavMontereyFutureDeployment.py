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
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
from HyperNav.Utilities.Data.WCOFS import WCOFSMonterey

file_handler = FilePathHandler(ROOT_DIR,'HypernavMontereyFutureDeployment')


class MontereyCartopy(RegionalBase):
    llcrnrlon=-122.4
    llcrnrlat=36.6
    urcrnrlon=-122
    urcrnrlat=36.8
    def __init__(self,*args,**kwargs):
        print('I am plotting Monterey')
        super().__init__(*args,**kwargs)

class WCOFSFutureMonterey(WCOFSMonterey):
	PlotClass = MontereyCartopy
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)


def monterey_mean_monthly_plot():
	uv_class = HYCOMFutureMonterey.load(datetime.datetime(2014,1,1),datetime.datetime(2022,1,1))
	mean_monthly_plot(uv_class,file_handler,month=12)


def monterey_quiver_movie():
	uv_class = WCOFSFutureMonterey.load()
	mask = [(x>datetime.datetime(2024,4,26,8)) for x in uv_class.time]
	quiver_movie(uv_class,mask,file_handler)


def monterey_shear_movie():
	uv_class = WCOFSFutureMonterey.load()
	lat = 36.7
	lon = -122.13
	mask = [(x>datetime.datetime(2024,4,26,8)) for x in uv_class.time]
	shear_movie(uv_class,mask,file_handler,lat,lon)

def monterey_eke():
	uv_class = WCOFSFutureMonterey.load()
	eke_plots(uv_class,file_handler)

def monterey_particles_compute():
	date_start = datetime.datetime(2024,4,29,8)
	date_end = datetime.datetime(2024,4,30,8)
	uv_class = WCOFSFutureMonterey.load()

	start_time = date_start.timestamp()
	end_time = date_end.timestamp()
	uv_class.depths[0]=0
	# uv_class = uv_class.subsample_depth(4,max_depth=-650)
	# uv_class = uv_class.subsample_time_u_v(3)
	data,dimensions = uv_class.return_parcels_uv(date_start-datetime.timedelta(hours=2),date_end+datetime.timedelta(hours=2))
	lat = 36.7
	lon = -122.13
	surface_time = 900
	vertical_speed = 0.076
	for depth in [50,100,200,300,400,500,600]:
		argo_cfg = {'lat': lat, 'lon': lon, 'target_lat': np.nan, 'target_lon': np.nan,
					'time': start_time, 'end_time': end_time, 'depth': 10, 'min_depth': 10, 'drift_depth': abs(depth),
					'max_depth': abs(depth),
					'surface_time': surface_time, 'total_cycle_time': 8*3600,
					'vertical_speed': vertical_speed,
					}
		create_prediction(argo_cfg,data,dimensions,'temp'+str(depth)+'.zarr',n_particles=500)		
	loc_dict = {}
	for depth in [50,100,200,300,400,500,600]:
		nc = ParticleDataset('temp'+str(depth)+'.zarr')
		dist_loc = []
		for delta in [datetime.timedelta(days=0,seconds=28800*x) for x in range(4)]:
			lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
			dist_loc.append((lat_center,lon_center))
		loc_dict[depth]=dist_loc
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = uv_class.plot(ax=ax)
	ax1.scatter(-122.186,36.712,s=100,label = 'MARS')
	ax1.plot(lon,lat)
	for depth in [50,100,200,300,400,500,600]:
		lats,lons = zip(*loc_dict[depth])
		ax1.scatter(lons,lats,label=str(depth)+' m')
		ax1.scatter(lons[-1],lats[-1],marker='*',color='k')
		ax1.plot(lons,lats)
	ax1.legend()