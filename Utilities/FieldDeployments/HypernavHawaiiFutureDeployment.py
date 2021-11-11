from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
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

def clear_days_prediction():
	uv_class = HYCOMFutureHawaii.load()
	float_list = [({'lat':19.5,'lon':-156.4,'time':datetime.datetime(2015,11,15)},'site_1')]
	dict_list = []
	for month,filename in [(12,'percent_clear_days_Dec_Hawaii.h5'),(1,'percent_clear_days_Jan_Hawaii.h5')]:
		dict_list.append((month,ClearSky(filename)))
	clear_sky_dict = dict(dict_list)
	pl = ParticleList()
	for float_pos_dict,filename in float_list:
		uv_class.time.set_ref_date(float_pos_dict['time'])
		for start_day in [5]*7:
			float_pos_dict['time'] = float_pos_dict['time']+datetime.timedelta(days=start_day)
			data,dimensions = uv_class.return_parcels_uv(float_pos_dict['time'],days_delta=30)
			prediction = UVPrediction(float_pos_dict,data,dimensions)
			prediction.create_prediction(ArgoVerticalMovement600,days=29.)
			nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
			pl.append(nc)
		for k,timedelta in enumerate([datetime.timedelta(days=x) for x in range(27)]):
			XX,YY,ax = uv_class.plot()
			pl.plot_density(timedelta,[uv_class.lons,uv_class.lats],ax)
			plt.savefig(file_handler.out_file('pdf_movie_'+filename+'/'+str(k)))
			plt.close()
		os.chdir(file_handler.out_file('pdf_movie_'+filename+'/'))
		os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")
		clear_sky_plot = []
		for time_delta in [datetime.timedelta(days = x ) for x in np.arange(2,30,2).tolist()]:
			lats,lons = pl.get_cloud_snapshot(time_delta)
			time = pl.get_time(time_delta)
			clear_sky_holder = [clear_sky_dict[12].return_clear_sky(y,x) for x,y,t in zip(lons,lats,time)]
			clear_sky_plot.append((time_delta.days,np.nanmean(clear_sky_holder)))
		days,sky = zip(*clear_sky_plot)
		Y = np.cumsum(np.array(sky)*.02)
		plt.plot(days,Y)
		plt.xlabel('Days')
		plt.ylabel('Cumulative Chance of Clear Sky')
		plt.savefig('cumulative_clear_sky')
		plt.close()
		plt.plot(days,sky)
		plt.xlabel('Days')
		plt.ylabel('Chance of Clear Sky')
		plt.savefig('clear_sky')
		plt.close()