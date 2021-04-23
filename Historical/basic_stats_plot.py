from hypernav.data_parse import load_files
from transition_matrix.makeplots.plot_utils import basemap_setup
import matplotlib.pyplot as plt
import numpy as np
from hypernav.data_parse import processed_base_file
from compute_utilities.list_utilities import find_nearest

def mean_plots()
	depth,lat,lon,time,uarray,varray = load_files()

	for k,d in enumerate(depth):

		u_mean = np.nanmean(uarray[:,k,:,:],axis=0)
		v_mean = np.nanmean(varray[:,k,:,:],axis=0)

		speed = np.sqrt(uarray[:,k,:,:]**2+varray[:,k,:,:]**2)

		total_std = np.nanstd(speed,axis=0)

		XX,YY,m = basemap_setup(lat,lon,'Moby')
		m.contourf(XX,YY,total_std)
		plt.colorbar(label='Speed Standard Deviation $m$ $s^{-1}$')
		QV = m.quiver(XX[::4,::4],YY[::4,::4],u_mean[::4,::4],v_mean[::4,::4])  
		if k in range(15):
			scale = 0.5
		else:
			scale = 0.1
		qk= plt.quiverkey (QV,0.95, 1.02, scale, str(scale)+' m/s')
		plt.title('Depth = '+str(d))
		plt.savefig(processed_base_file+'plots/mean_'+str(int(float(d))))
		plt.close()

def shear_plots():
	def return_from_idx(ii,jj):
		u_mean = np.nanmean(uarray[:,:,ii,jj],axis=0)
		v_mean = np.nanmean(varray[:,:,ii,jj],axis=0)
		speed = np.sqrt(uarray[:,:,ii,jj]**2+varray[:,:,ii,jj]**2)
		mean_speed = np.nanmean(speed,axis=0)
		return (u_mean,v_mean,mean_speed)

	depth,lat,lon,time,uarray,varray = load_files()


	center_lat = 20.8
	center_lon = -157.2

	plt.subplot(2,1,1)
	from plot_utilities.eulerian_plot import Basemap
	llcrnrlon=(center_lon-0.5)
	llcrnrlat=(center_lat-0.5)
	urcrnrlon=(center_lon+0.5)
	urcrnrlat=(center_lat+0.5)
	lon_0 = 0
	m = Basemap.auto_map(urcrnrlat,llcrnrlat,urcrnrlon,llcrnrlon,lon_0,aspect=True,resolution='h')
	m.scatter(center_lon,center_lat,150,marker='*',color='Red',latlon=True,zorder=10)
	lon_list = np.arange(center_lon-0.2,center_lon+0.21,0.1)
	lat_list = np.arange(center_lat-0.2,center_lat+0.21,0.1)

	XX,YY = np.meshgrid(lon_list,lat_list)
	m.scatter(XX.flatten(),YY.flatten(),latlon=True)

	nearest_lat = [find_nearest(lat,_) for _ in YY.flatten()]
	nearest_lon = [find_nearest(lon,_) for _ in XX.flatten()]

	lat_idx = [lat.tolist().index(_) for _ in nearest_lat]
	lon_idx = [lon.tolist().index(_) for _ in nearest_lon]

	for ii,jj in zip(lat_idx,lon_idx):
		u_mean,v_mean,mean_speed = return_from_idx(ii,jj)
		plt.subplot(2,3,4)
		plt.plot(u_mean,depth)
		plt.subplot(2,3,5)
		plt.plot(v_mean,depth)
		plt.subplot(2,3,6)
		plt.plot(mean_speed,depth)
	plt.subplot(2,3,4)
	plt.ylim([300,700])
	plt.xlabel('m/s')
	plt.gca().invert_yaxis()
	plt.ylabel('Depth (m)')
	plt.title('Mean U')
	plt.subplot(2,3,5)
	frame1 = plt.gca()
	plt.xlabel('m/s')
	plt.ylim([300,1200])
	frame1.invert_yaxis()
	frame1.axes.get_yaxis().set_visible(False)
	plt.title('Mean V')	
	plt.subplot(2,3,6)
	frame2 = plt.gca()
	plt.xlabel('m/s')
	plt.ylim([300,1200])
	frame2.invert_yaxis()
	frame2.axes.get_yaxis().set_visible(False)
	plt.title('Mean Speed')
	plt.savefig(processed_base_file+'plots/profile_velocity_plots')
	plt.close()

