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
from HyperNav.Utilities.Data.HYCOM import HYCOMSouthernCalifornia
file_handler = FilePathHandler(ROOT_DIR,'HypernavSoCalFutureDeployment')
from HyperNav.Utilities.Compute.__init__ import ROOT_DIR as COMPUTE_DIR
compute_file_handler = FilePathHandler(COMPUTE_DIR,'RunParcels')
import gc

class SoCalCartopy(RegionalBase):
    llcrnrlon=-120
    llcrnrlat=33
    urcrnrlon=-118
    urcrnrlat=35
    def __init__(self,*args,**kwargs):
        print('I am plotting Southern California')
        super().__init__(*args,**kwargs)

class HYCOMFutureSoCal(HYCOMSouthernCalifornia):
	PlotClass = SoCalCartopy
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)


def socal_mean_monthly_plot():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2021,1,20),datetime.datetime(2021,2,28))
	mean_monthly_plot(uv_class,file_handler,month=12)


def socal_quiver_movie():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2023,3,1),datetime.datetime(2023,4,20))
	mask = [(x>datetime.datetime(2023,4,1))&(x<datetime.datetime(2023,4,18)) for x in uv_class.time]
	quiver_movie(uv_class,mask,file_handler)


def socal_point_time_series_movie():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2023,3,1),datetime.datetime(2023,4,20))
	mask = [(x>datetime.datetime(2023,4,1))&(x<datetime.datetime(2023,4,18)) for x in uv_class.time]
	lat = 33.3382
	lon = -119.3411
	depth = -700
	time_series_movie(uv_class,mask,file_handler,lat,lon,depth)


def socal_shear_movie():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2023,3,1),datetime.datetime(2023,4,20))
	mask = [(x>datetime.datetime(2023,4,1))&(x<datetime.datetime(2023,4,18)) for x in uv_class.time]
	lat = 33.3382
	lon = -119.3411
	shear_movie(uv_class,mask,file_handler,lat,lon)

def socal_eke():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2014,1,1),datetime.datetime(2022,1,1))
	eke_plots(uv_class,file_handler)

def monterey_particles_compute():
	uv_class = HYCOMFutureSoCal
	float_list = [({'lat':32.9,'lon':-117.8,'time':datetime.datetime(2015,11,20)},'site_1')]
	pdf_particles_compute(uv_class,float_list,file_handler)

def SoCalParticlesCompute():
	date_start = datetime.datetime(2022,1,20)
	date_end = datetime.datetime(2022,2,5)
	uv_class = HYCOMFutureSoCal.load(date_start-datetime.timedelta(days=3),date_end+datetime.timedelta(days=3))

	start_time = date_start.timestamp()
	end_time = date_end.timestamp()
	uv_class.depths[0]=0
	# uv_class = uv_class.subsample_depth(4,max_depth=-650)
	# uv_class = uv_class.subsample_time_u_v(3)
	data,dimensions = uv_class.return_parcels_uv(date_start-datetime.timedelta(days=2),date_end+datetime.timedelta(days=2))
	lat = 33.75
	lon = -119.5
	surface_time = 5400
	vertical_speed = 0.076
	for depth in [300,400,500,600]:
		argo_cfg = {'lat': lat, 'lon': lon, 'target_lat': np.nan, 'target_lon': np.nan,
					'time': start_time, 'end_time': end_time, 'depth': 10, 'min_depth': 10, 'drift_depth': abs(depth),
					'max_depth': abs(depth),
					'surface_time': surface_time, 'total_cycle_time': 24*3600,
					'vertical_speed': vertical_speed,
					}

		dist_loc = []
		filename = 'site_1_'+str(depth)
		if not os.path.isdir(compute_file_handler.tmp_file(filename+'.zarr')):
			create_prediction(argo_cfg,data,dimensions,compute_file_handler.tmp_file(filename))		
		nc = ParticleDataset(compute_file_handler.tmp_file(filename+'.zarr'))
		for delta in [datetime.timedelta(days=x,seconds=0) for x in range(14)]:
			lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
			dist_loc.append((lat_center,lon_center))
		y,x = nc.total_coords()
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		XX,YY,ax1 = uv_class.plot(ax=ax1)
		for n in range(x.shape[0]):
			ax1.scatter(x[n,:],y[n,:],s=0.2)
			ax1.scatter(x[n,-1],y[n,-1],s=1,c='k',marker='x',zorder=20)
		plt.title(filename)
		plt.savefig(file_handler.out_file(filename))
		plt.close()	
		data_list = []
		z = np.ma.masked_array(nc.zarr_load('z'))
		for n in range(y.shape[0]):
			data_list.append(len(z[n,:][~z[n,:].mask].data.ravel()))
		plt.hist(data_list,bins=100)
		plt.xlabel('Time Step')
		plt.title('Timestep Ran Aground')
		plt.savefig(file_handler.out_file(filename+'_hist'))
		plt.close()			
		gc.collect(generation=2)
