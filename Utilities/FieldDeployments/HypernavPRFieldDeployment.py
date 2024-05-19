from HyperNav.Utilities.Data.HYCOM import HYCOMPuertoRico
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
import numpy as np
import os
file_handler = FilePathHandler(ROOT_DIR,'HypernavPuertoRicoFieldDeployment')
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
import math


class HRPuertoRicoCartopy(RegionalBase):
    llcrnrlon=-68.5 
    llcrnrlat=16.5
    urcrnrlon=-65
    urcrnrlat=18.5
    def __init__(self,*args,**kwargs):
        print('I am plotting Puerto Rico')
        super().__init__(*args,**kwargs)

HYCOMPuertoRico.PlotClass = HRPuertoRicoCartopy


def PRParticlesCompute():
	uv_class = HYCOMPuertoRico.load()
	for float_pos_dict,filename in [({'lat':18.6,'lon':-67.75,'time':datetime.datetime(2015,12,1)},'site_1'),
		({'lat':18.8,'lon':-66.75,'time':datetime.datetime(2015,12,1)},'site_2')]:
		dist_loc = []
		for start_day in [1]*10:
			float_pos_dict['time'] = float_pos_dict['time']+datetime.timedelta(days=start_day)
			data,dimensions = uv.return_parcels_uv(float_pos_dict['time'],days_delta=15)
			prediction = UVPrediction(float_pos_dict,data,dimensions)
			prediction.create_prediction(ArgoVerticalMovement600,days=14.)
			nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
			for delta in [datetime.timedelta(days=x) for x in [14]]:
				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
				dist_loc.append((lat_center,lon_center))		
		dist_lat,dist_lon = zip(*dist_loc)
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		XX,YY,ax1 = uv.plot(ax=ax1)
		ax1.scatter(dist_lon,dist_lat)
		lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
		ax1.scatter(lon_center,lat_center)
		ax1.scatter(float_pos_dict['lon'],float_pos_dict['lat'],marker='x',c='k',linewidth=6,s=250,)
		plt.savefig(file_handler.out_file(filename))
		plt.close()


def mean_monthly_plot():
	holder = HYCOMPuertoRico.load(datetime.datetime(2024,4,1),datetime.datetime(2024,4,15))
	fig = plt.figure(figsize=(12,7))
	ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree())
	ax4 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
	XX,YY,ax1 = holder.plot(ax=ax1)
	u,v = holder.return_monthly_mean(4,-600)
	q = ax1.quiver(XX,YY,u,v,scale=5)
	ax1.title.set_text('Depth = 600m')
	XX,YY,ax2 = holder.plot(ax=ax2)
	u,v = holder.return_monthly_mean(4,-200)
	q = ax2.quiver(XX,YY,u,v,scale=5)
	ax2.title.set_text('Depth = 200m')
	XX,YY,ax3 = holder.plot(ax=ax3)
	u,v = holder.return_monthly_mean(4,-50)
	q = ax3.quiver(XX,YY,u,v,scale=5)
	ax3.title.set_text('Depth = 50m')
	XX,YY,ax4 = holder.plot(ax=ax4)
	u,v = holder.return_monthly_mean(4,0)
	q = ax4.quiver(XX,YY,u,v,scale=5)
	ax4.title.set_text('Depth = Surface')
	plt.savefig(file_handler.out_file('monthly_mean_quiver'))
	plt.close()

def quiver_movie():
	holder = HYCOMPuertoRico.load(datetime.datetime(2024,4,1),datetime.datetime(2024,4,15))
	shallow = 0
	deep = -500
	u = holder.u[:,:,:,:]
	v = holder.v[:,:,:,:]
	time = np.array(holder.time)
	deep_idx = holder.depths.find_nearest(deep,idx=True)
	shallow_idx = holder.depths.find_nearest(shallow,idx=True)
	for k in range(u.shape[0]):
		u_holder = u[k,:,:,:]
		v_holder = v[k,:,:,:]
		fig = plt.figure(figsize=(12,7))
		ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
		XX,YY,ax1 = holder.plot(ax=ax1)
		ax1.quiver(XX,YY,u_holder[deep_idx,:,:],v_holder[deep_idx,:,:],scale=7)
		ax1.title.set_text('Depth = 600m')
		ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
		XX,YY,ax2 = holder.plot(ax=ax2)
		q = ax2.quiver(XX,YY,u_holder[shallow_idx,:,:],v_holder[shallow_idx,:,:],scale=7)
		ax2.quiverkey(q,X=-0.3, Y=1.02, U=1,
             label='Quiver key, length = 1 m/s', labelpos='E')
		ax2.title.set_text('Depth = Surface')
		plt.suptitle(time[k].ctime())
		plt.savefig(file_handler.out_file('quiver_movie/'+str(k)))
		plt.close()
	os.chdir(file_handler.out_file('quiver_movie/'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")

def shear_movie():
	holder = HYCOMPuertoRico.load(datetime.datetime(2024,4,1),datetime.datetime(2024,4,15))
	lat = 17.75
	lon = -66.6
	mask = [(x>datetime.datetime(2024,4,1))&(x<datetime.datetime(2024,4,15)) for x in holder.time]
	time = np.array(holder.time)[mask]
	depths = holder.depths[:(holder.u.shape[1])]
	for k,t in enumerate(time):
		fig = plt.figure(figsize=(12,12))
		u,v = holder.vertical_shear(t,lat,lon)

		ax1 = fig.add_subplot(1,2,1)
		ax1.plot(u,depths,label='u')
		ax1.plot(v,depths,label='v')
		ax1.set_xlim([-0.55,0.55])
		ax1.set_xlabel('Current Speed $ms^{-1}$')
		ax1.set_ylabel('Depth (m)')
		
		ax2 = fig.add_subplot(2,2,2, polar=True)
		ax2.set_rlim([0,0.9])
		for u_holder,v_holder in zip(u,v):
			theta = math.atan2(v_holder,u_holder)
			r = np.sqrt(u_holder**2+v_holder**2)
			ax2.arrow(theta,(r*0.5),0,r,alpha = 0.5, width = 0.06,
                 edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)
		ax3 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
		XX,YY,ax3 = holder.plot(ax=ax3)
		ax3.scatter(lon,lat,s=100,zorder=10)
		u,v = holder.return_u_v(time=t,depth=0)
		ax3.quiver(XX,YY,u,v,scale=7)
		plt.suptitle(t.ctime())
		plt.savefig(file_handler.out_file('shear_movie/'+str(k)))
		plt.close()
	os.chdir(file_handler.out_file('shear_movie/'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")

def eke():
	holder = HYCOMPuertoRico.load()
	shallow = 0
	deep = -600
	shallow_idx = holder.depth.find_nearest(shallow,idx=True)
	deep_idx = holder.depth.find_nearest(deep,idx=True)

	u_mean = np.nanmean(holder.u[:,:,:,:],axis=0)
	u_mean = np.stack([u_mean]*holder.u.shape[0])
	v_mean = np.nanmean(holder.v[:,:,:,:],axis=0)
	v_mean = np.stack([v_mean]*holder.v.shape[0])
	u = np.nanmean((holder.u-u_mean)**2,axis=0)
	v = np.nanmean((holder.v-v_mean)**2,axis=0)

	eke_deep = u[deep_idx,:,:]+v[deep_idx,:,:]
	eke_shallow = u[shallow_idx,:,:]+v[shallow_idx,:,:]

	fig = plt.figure(figsize=(12,7))
	ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
	ax1.title.set_text('Depth = 400m')
	ax2.title.set_text('Depth = Surface')
	XX,YY,ax1 = holder.plot(ax=ax1)
	ax1.pcolor(XX,YY,eke_deep,vmax = (np.nanmean(eke_shallow)+2*np.nanstd(eke_shallow)))
	XX,YY,ax2 = holder.plot(ax=ax2)
	ax2.pcolor(XX,YY,eke_shallow,vmax = (np.nanmean(eke_shallow)+2*np.nanstd(eke_shallow)))
	PCM = ax2.get_children()[0]
	fig.colorbar(PCM,ax=[ax1,ax2],pad=.05,label='Eddy Kinetic Energy ($m^2 s^{-2}$)',location='bottom')
	plt.savefig(file_handler.out_file('eke_plot'))
	plt.close()
