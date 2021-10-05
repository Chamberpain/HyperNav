from HyperNav.Utilities.Data.HYCOM import HYCOMTahiti
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import bathy_plot,mean_monthly_plot,quiver_movie,shear_movie,eke_plots
from HyperNav.Utilities.Compute.RunParcels import UVPrediction,ParticleDataset
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50

import cartopy.crs as ccrs
import numpy as np
import os
file_handler = FilePathHandler(ROOT_DIR,'HypernavTahitiFieldDeployment')


def tahiti_bathy_plot():
	uv_class = HYCOMTahiti.recompile_hindcast([2015])
	bathy_plot(uv_class,file_handler)

def tahiti_mean_monthly_plot():
	uv_class = HYCOMTahiti.recompile_hindcast([2013,2014,2015])
	mean_monthly_plot(uv_class,file_handler,month=6)


def tahiti_quiver_movie():
	uv_class = HYCOMTahiti.recompile_hindcast([2015])
	mask = [(x>datetime.datetime(2015,6,1))&(x<datetime.datetime(2015,6,30)) for x in uv_class.time]
	quiver_movie(uv_class,mask,file_handler)


def tahiti_shear_movie():
	uv_class = HYCOMTahiti.recompile_hindcast([2015])
	lat = -17.5
	lon = -151
	mask = [(x>datetime.datetime(2015,6,1))&(x<datetime.datetime(2015,6,30)) for x in uv_class.time]
	shear_movie(uv_class,mask,file_handler,lat,lon)

def tahiti_eke():
	uv_class = HYCOMTahiti.recompile_hindcast([2015])
	eke_plots(uv_class,file_handler)


def TahitiParticlesCompute():
	uv_class = HYCOMTahiti.recompile_hindcast([2015])
	float_list = [({'lat':-17.5,'lon':-151,'time':datetime.datetime(2015,6,1)},'site_1')]
	uv_class.time.set_ref_date(datetime.datetime(2015,6,1))
	for float_pos_dict,filename in float_list:
		dist_loc = []
		for start_day in [5]*3:
			float_pos_dict['time'] = float_pos_dict['time']+datetime.timedelta(days=start_day)
			data,dimensions = uv_class.return_parcels_uv(float_pos_dict['time'],days_delta=15)
			prediction = UVPrediction(float_pos_dict,data,dimensions)
			prediction.create_prediction(ArgoVerticalMovement600,days=14.)
			nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
			for delta in [datetime.timedelta(days=x) for x in [14]]:
				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
				dist_loc.append((lat_center,lon_center))		
		dist_lat,dist_lon = zip(*dist_loc)
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		XX,YY,ax1 = uv_class.plot(ax=ax1)
		ax1.scatter(dist_lon,dist_lat)
		lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
		ax1.scatter(lon_center,lat_center)
		ax1.scatter(float_pos_dict['lon'],float_pos_dict['lat'],marker='x',c='k',linewidth=6,s=250,)
		plt.savefig(file_handler.out_file(filename))
		plt.close()
