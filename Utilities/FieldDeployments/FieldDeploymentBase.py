import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import math

def bathy_plot(uv_class,file_handler):
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
	X,Y,ax = uv_class.plot(ax=ax)
	cf = ax.bathy()
	fig.colorbar(cf,label='Depth (km)')
	plt.savefig(file_handler.out_file('depth'))

def mean_monthly_plot(uv_class,file_handler,month=None):
	fig = plt.figure(figsize=(12,7))
	ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree())
	ax4 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
	XX,YY,ax1 = uv_class.plot(ax=ax1)
	u,v = uv_class.return_monthly_mean(month,-600)
	q = ax1.quiver(XX,YY,u,v,scale=5)
	ax1.title.set_text('Depth = 600m')
	XX,YY,ax2 = uv_class.plot(ax=ax2)
	u,v = uv_class.return_monthly_mean(month,-200)
	q = ax2.quiver(XX,YY,u,v,scale=5)
	ax2.title.set_text('Depth = 200m')
	XX,YY,ax3 = uv_class.plot(ax=ax3)
	u,v = uv_class.return_monthly_mean(month,-50)
	q = ax3.quiver(XX,YY,u,v,scale=5)
	ax3.title.set_text('Depth = 50m')
	XX,YY,ax4 = uv_class.plot(ax=ax4)
	u,v = uv_class.return_monthly_mean(month,0)
	q = ax4.quiver(XX,YY,u,v,scale=5)
	ax4.title.set_text('Depth = Surface')
	plt.savefig(file_handler.out_file('monthly_mean_quiver'))
	plt.close()

def quiver_movie(uv_class,mask,file_handler):
	shallow = 0
	deep = -600
	u = uv_class.u[mask,:,:,:]
	v = uv_class.v[mask,:,:,:]
	time = np.array(uv_class.time)[mask]
	deep_idx = uv_class.depth.find_nearest(deep,idx=True)
	shallow_idx = uv_class.depth.find_nearest(shallow,idx=True)
	for k in range(u.shape[0]):
		u_uv_class = u[k,:,:,:]
		v_uv_class = v[k,:,:,:]
		fig = plt.figure(figsize=(12,7))
		ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
		XX,YY,ax1 = uv_class.plot(ax=ax1)
		ax1.quiver(XX,YY,u_uv_class[deep_idx,:,:],v_uv_class[deep_idx,:,:],scale=7)
		ax1.title.set_text('Depth = 600m')
		ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
		XX,YY,ax2 = uv_class.plot(ax=ax2)
		q = ax2.quiver(XX,YY,u_uv_class[shallow_idx,:,:],v_uv_class[shallow_idx,:,:],scale=7)
		ax2.quiverkey(q,X=-0.3, Y=1.02, U=1,
             label='Quiver key, length = 1 m/s', labelpos='E')
		ax2.title.set_text('Depth = Surface')
		plt.suptitle(time[k].ctime())
		plt.savefig(file_handler.out_file('quiver_movie/'+str(k)))
		plt.close()
	os.chdir(file_handler.out_file('quiver_movie/'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")

def shear_movie(uv_class,mask,file_handler,lat,lon):
	time = np.array(uv_class.time)[mask]
	depths = uv_class.depth[:(uv_class.u.shape[1])]
	for k,t in enumerate(time):
		fig = plt.figure(figsize=(12,12))
		u,v = uv_class.vertical_shear(t,lat,lon)

		ax1 = fig.add_subplot(1,2,1)
		ax1.plot(u,depths,label='u')
		ax1.plot(v,depths,label='v')
		ax1.set_xlim([-0.55,0.55])
		ax1.set_xlabel('Current Speed $ms^{-1}$')
		ax1.set_ylabel('Depth (m)')
		
		ax2 = fig.add_subplot(2,2,2, polar=True)
		ax2.set_rlim([0,0.9])
		for u_uv_class,v_uv_class in zip(u,v):
			theta = math.atan2(v_uv_class,u_uv_class)
			r = np.sqrt(u_uv_class**2+v_uv_class**2)
			ax2.arrow(theta,(r*0.5),0,r,alpha = 0.5, width = 0.06,
                 edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)
		ax3 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
		XX,YY,ax3 = uv_class.plot(ax=ax3)
		ax3.scatter(lon,lat,s=100,zorder=10)
		u,v = uv_class.return_u_v(time=t,depth=0)
		ax3.quiver(XX,YY,u,v,scale=7)
		plt.suptitle(t.ctime())
		plt.savefig(file_handler.out_file('shear_movie/'+str(k)))
		plt.close()
	os.chdir(file_handler.out_file('shear_movie/'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")

def pdf_particles_compute(uv_class,float_list,file_handler):
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
		plt.savefig(file_handler.out_file('pdf_movie/'+str(k)))
		plt.close()
	os.chdir(file_handler.out_file('pdf_movie/'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")


def eke_plots(uv_class,file_handler):
	shallow = 0
	deep = -600
	shallow_idx = uv_class.depth.find_nearest(shallow,idx=True)
	deep_idx = uv_class.depth.find_nearest(deep,idx=True)

	u_mean = np.nanmean(uv_class.u[:,:,:,:],axis=0)
	u_mean = np.stack([u_mean]*uv_class.u.shape[0])
	v_mean = np.nanmean(uv_class.v[:,:,:,:],axis=0)
	v_mean = np.stack([v_mean]*uv_class.v.shape[0])
	u = np.nanmean((uv_class.u-u_mean)**2,axis=0)
	v = np.nanmean((uv_class.v-v_mean)**2,axis=0)

	eke_deep = u[deep_idx,:,:]+v[shallow_idx,:,:]
	eke_shallow = u[shallow_idx,:,:]+v[shallow_idx,:,:]

	fig = plt.figure(figsize=(12,7))
	ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
	ax1.title.set_text('Depth = 600m')
	ax2.title.set_text('Depth = Surface')
	XX,YY,ax1 = uv_class.plot(ax=ax1)
	ax1.pcolor(XX,YY,eke_deep,vmax = (eke_shallow.mean()+2*eke_shallow.std()))
	XX,YY,ax2 = uv_class.plot(ax=ax2)
	ax2.pcolor(XX,YY,eke_shallow,vmax = (eke_shallow.mean()+2*eke_shallow.std()))
	PCM = ax2.get_children()[0]
	fig.colorbar(PCM,ax=[ax1,ax2],pad=.05,label='Eddy Kinetic Energy ($m^2 s^{-2}$)',location='bottom')
	plt.savefig(file_handler.out_file('eke_plot'))
	plt.close()