from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
file_handler = FilePathHandler(ROOT_DIR,'HypernavHawaiiFieldDeployment')


def mean_monthly_plot():
	holder = HYCOMHawaii.load()
	fig = plt.figure(figsize=(12,7))
	ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree())
	ax4 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
	XX,YY,ax1 = holder.plot(ax=ax1)
	u,v = holder.return_monthly_mean(6,-600)
	q = ax1.quiver(XX,YY,u,v,scale=5)
	ax1.title.set_text('Depth = 600m')
	XX,YY,ax2 = holder.plot(ax=ax2)
	u,v = holder.return_monthly_mean(6,-200)
	q = ax2.quiver(XX,YY,u,v,scale=5)
	ax2.title.set_text('Depth = 200m')
	XX,YY,ax3 = holder.plot(ax=ax3)
	u,v = holder.return_monthly_mean(6,-50)
	q = ax3.quiver(XX,YY,u,v,scale=5)
	ax3.title.set_text('Depth = 50m')
	XX,YY,ax4 = holder.plot(ax=ax4)
	u,v = holder.return_monthly_mean(6,0)
	q = ax4.quiver(XX,YY,u,v,scale=5)
	ax4.title.set_text('Depth = Surface')
	plt.savefig(file_handler.out_file('monthly_mean_quiver'))
	plt.close()

def quiver_movie():
	holder = HYCOMHawaii.load()
	shallow = 0
	deep = -600
	mask = [(x>datetime.datetime(2021,6,6))&(x<datetime.datetime(2021,6,24)) for x in holder.time]
	u = holder.u[mask,:,:,:]
	v = holder.v[mask,:,:,:]
	time = np.array(holder.time)[mask]
	deep_idx = holder.depth.find_nearest(deep,idx=True)
	shallow_idx = holder.depth.find_nearest(shallow,idx=True)
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
	os.system("ffmpeg -r 1 -i %01d.png -vcodec mpeg4 -y movie.mp4")

def eke_movie():
	holder = HYCOMHawaii.load()
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

	fig = plt.figure(figsize=(12,7))
	ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
	XX,YY,ax1 = holder.plot(ax=ax1)
	ax1.pcolor(XX,YY,u[deep_idx,:,:]+v[shallow_idx,:,:])
	XX,YY,ax2 = holder.plot(ax=ax2)
	ax2.pcolor(XX,YY,u[shallow_idx,:,:]+v[shallow_idx,:,:])


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
	os.system("ffmpeg -r 1 -i %01d.png -vcodec mpeg4 -y movie.mp4")
