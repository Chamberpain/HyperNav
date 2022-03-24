from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots,pdf_particles_compute
from HyperNav.Utilities.Compute.RunParcels import UVPrediction,ParticleDataset
import cartopy.crs as ccrs
import numpy as np
import os
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
from HyperNav.Utilities.FieldDeployments.HypernavHawaiiFutureDeployment import add_zero_boundary_conditions
from HyperNav.Utilities.Data.HYCOM import HYCOMBase
from GeneralUtilities.Data.depth.depth_utilities import ETopo1Depth
file_handler = FilePathHandler(ROOT_DIR,'HypernavGOMDeployment')
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList,flat_list
from HyperNav.Utilities.Data.UVBase import UVTimeList
from HyperNav.Utilities.Compute.RunParcels import ParticleList,UVPrediction,ParticleDataset

def ArgoVerticalMovement(particle, fieldset, time):
	driftdepth = 1500  # maximum depth in m
	maxdepth = 2000  # maximum depth in m
	vertical_speed = 0.10  # sink and rise speed in m/s
	surftime = 0.5 * 3600  # time of deep drift in seconds
	cycletime = 10 * 86400-(2000-driftdepth)/vertical_speed-2000/vertical_speed-surftime  # total time of cycle in seconds
	mindepth = 10

	if particle.cycle_phase == 0:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= driftdepth:
			particle.cycle_phase = 1

	elif particle.cycle_phase == 1:
		# Phase 1: Drifting at depth for drifttime seconds
		particle.cycle_age += particle.dt
		if particle.cycle_age >= cycletime:
			particle.cycle_phase = 2

	if particle.cycle_phase == 2:
		# Phase 0: Sinking with vertical_speed until depth is driftdepth
		particle.depth += vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth >= 2000:
			particle.cycle_phase = 3


	elif particle.cycle_phase == 3:
		# Phase 3: Rising with vertical_speed until at surface
		particle.depth -= vertical_speed * particle.dt
		particle.cycle_age += particle.dt
		if particle.depth <= mindepth:
			particle.depth = mindepth
			particle.surf_age = 0
			particle.cycle_phase = 4

	elif particle.cycle_phase == 4:
		# Phase 4: Transmitting at surface until cycletime is reached
		particle.cycle_age += particle.dt
		particle.surf_age += particle.dt
		if particle.surf_age > surftime:
			particle.cycle_phase = 0
			particle.cycle_age = 0  # reset cycle_age for next cycle






class GOMCartopy(RegionalBase):
	llcrnrlon=-87
	llcrnrlat=23
	urcrnrlon=-84
	urcrnrlat=25
	def __init__(self,*args,**kwargs):
		print('I am plotting GOM')
		super().__init__(*args,**kwargs)


class HYCOMGOM(HYCOMBase):
	location='GOM'
	urlat = 25
	lllat = 20
	lllon = -86
	urlon = -82
	ID = 'HYCOM_reg1_latest3d'
	PlotClass = GOMCartopy
	DepthClass = ETopo1Depth
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.u = add_zero_boundary_conditions(self.u)
		self.v = add_zero_boundary_conditions(self.v)	

	@classmethod
	def get_dimensions(cls):
		from pydap.client import open_url
		dataset = open_url(cls.base_html+cls.ID)
		#open dataset using erdap server
		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%d-%b-%Y %H:%M:%S')
		UVTimeList.set_ref_date(time_since)
		time = UVTimeList.time_list_from_seconds(dataset['time'][:])
		lats = LatList(dataset['latitude'][:])
		lons = dataset['longitude'][:].data
		lons[lons>180] = lons[lons>180]-360
		lons = LonList(lons)
		depth = -dataset['depth'][:].data
		depth = DepthList(depth)
		depth_idx = depth.find_nearest(-2500,idx=True)
		depth = depth[:(depth_idx+1)]
		higher_lon_idx = lons.find_nearest(cls.urlon,idx=True)
		lower_lon_idx = lons.find_nearest(cls.lllon,idx=True)
		lons = lons[lower_lon_idx:higher_lon_idx]
		higher_lat_idx = lats.find_nearest(cls.urlat,idx=True)
		lower_lat_idx = lats.find_nearest(cls.lllat,idx=True)
		lats = lats[lower_lat_idx:higher_lat_idx]
		units = dataset['water_u'].attributes['units']
		return (time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,units)

date_start = datetime.datetime(2020,11,1)
date_end = datetime.datetime(2020,12,1)


def gom_mean_monthly_plot():
	uv_class = HYCOMGOM.load(date_start,date_end)
	month=11
	fig = plt.figure(figsize=(12,7))
	ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
	ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
	ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree())
	ax4 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
	XX,YY,ax1 = uv_class.plot(ax=ax1)
	u,v = uv_class.return_monthly_mean(month,-2000)
	q = ax1.quiver(XX,YY,u,v,scale=5)
	ax1.plot(lon_list,lat_list,'r-*')
	ax1.title.set_text('Depth = 2000m')
	XX,YY,ax2 = uv_class.plot(ax=ax2)
	ax2.plot(lon_list,lat_list,'r-*')
	u,v = uv_class.return_monthly_mean(month,-1500)
	q = ax2.quiver(XX,YY,u,v,scale=5)
	ax3.plot(lon_list,lat_list,'r-*')
	ax2.title.set_text('Depth = 1500m')
	XX,YY,ax3 = uv_class.plot(ax=ax3)
	u,v = uv_class.return_monthly_mean(month,-1250)
	q = ax3.quiver(XX,YY,u,v,scale=5)
	ax3.plot(lon_list,lat_list,'r-*')
	ax3.title.set_text('Depth = 1250m')
	XX,YY,ax4 = uv_class.plot(ax=ax4)
	u,v = uv_class.return_monthly_mean(month,-1000)
	q = ax4.quiver(XX,YY,u,v,scale=5)
	ax4.plot(lon_list,lat_list,'r-*')
	ax4.title.set_text('Depth = 1000m')
	plt.savefig(file_handler.out_file('monthly_mean_quiver'))
	plt.close()

def gom_quiver_movie():
	uv_class = HYCOMGOM.load(date_start,date_end)
	mask = [(x>datetime.datetime(2020,11,1))&(x<datetime.datetime(2020,12,1)) for x in uv_class.time]
	shallow = -1000
	deep = -1500
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
		ax1.title.set_text('Depth = '+str(deep))
		ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
		XX,YY,ax2 = uv_class.plot(ax=ax2)
		q = ax2.quiver(XX,YY,u_uv_class[shallow_idx,:,:],v_uv_class[shallow_idx,:,:],scale=7)
		ax2.quiverkey(q,X=-0.3, Y=1.02, U=1,
			 label='Quiver key, length = 1 m/s', labelpos='E')
		ax2.title.set_text('Depth = '+str(shallow))
		ax1.plot(lon_list,lat_list,'r-*')
		ax2.plot(lon_list,lat_list,'r-*')

		plt.suptitle(time[k].ctime())
		plt.savefig(file_handler.out_file('quiver_movie/'+str(k)))
		plt.close()
	os.chdir(file_handler.out_file('quiver_movie/'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")



def gom_shear_movie():
	uv_class = HYCOMGOM.load(date_start,date_end)
	lat = 24.662
	lon = -84.794
	mask = [(x>datetime.datetime(2020,11,1))&(x<datetime.datetime(2020,11,30)) for x in uv_class.time]
	shear_movie(uv_class,mask,file_handler,lat,lon)

def gom_eke():
	uv_class = HYCOMGOM.load(date_start,date_end)
	shallow = 0
	deep = -1500
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
	ax1.title.set_text('Depth = 1500m')
	ax2.title.set_text('Depth = Surface')
	XX,YY,ax1 = uv_class.plot(ax=ax1)
	ax1.pcolor(XX,YY,eke_deep,vmax = (eke_shallow.mean()+2*eke_shallow.std()))
	XX,YY,ax2 = uv_class.plot(ax=ax2)
	ax2.pcolor(XX,YY,eke_shallow,vmax = (eke_shallow.mean()+2*eke_shallow.std()))
	PCM = ax2.get_children()[0]
	ax1.plot(lon_list,lat_list,'r-*')
	ax2.plot(lon_list,lat_list,'r-*')
	fig.colorbar(PCM,ax=[ax1,ax2],pad=.05,label='Eddy Kinetic Energy ($m^2 s^{-2}$)',location='bottom')
	plt.savefig(file_handler.out_file('eke_plot'))
	plt.close()

def future_prediction():
	date_start = datetime.datetime(2022,3,3)
	date_end = datetime.datetime(2022,3,10)
	uv_class = HYCOMGOM.load(date_start-datetime.timedelta(days=1),date_end)
	lats = [23.861]
	lons = [-85.158]
	dates = [datetime.datetime(2022,3,3)]
	dates = [date_start]*len(lons)
	keys = ['lat','lon','time']
	float_list = [dict(zip(keys,list(x))) for x in zip(lats,lons,dates)]
	pl = ParticleList()
	for float_pos_dict in float_list:
		uv_class.time.set_ref_date(float_pos_dict['time'])
		data,dimensions = uv_class.return_parcels_uv(float_pos_dict['time']-datetime.timedelta(hours=1),days_delta=7)
		prediction = UVPrediction(float_pos_dict,data,dimensions)
		prediction.create_prediction(ArgoVerticalMovement600,days=4.5)
		nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
		pl.append(nc)
	plt.rcParams["figure.figsize"] = (15,15)
	lat_list = []
	lon_list = []
	for r,timedelta in enumerate([datetime.timedelta(hours=int(x)) for x in range(int(24*4.5))[::3]]):
		scatter_list = [x.get_cloud_center(timedelta) for x in pl]
		lat,lon,lat_std,lon_std = zip(*scatter_list)
		lat_list.append(list(lat))
		lon_list.append(list(lon))
		XX,YY,ax = uv_class.plot()
		ax.scatter(lon,lat,s=80,marker='X',zorder=15)

		lat_holder = np.vstack(lat_list)
		lon_holder = np.vstack(lon_list)
		for k in range(lat_holder.shape[1]):
			ax.plot(lon_holder[:,k],lat_holder[:,k],'b',alpha=0.8,linewidth=3)
		plt.title(date_start+timedelta)
		plt.savefig(file_handler.out_file('deployment_movie/'+str(r)))
		plt.close()
	os.chdir(file_handler.out_file('deployment_movie'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")






def GOM_particles_compute():
	uv_class = HYCOMGOM
	float_list = [({'lat':24.662,'lon':-84.794,'time':datetime.datetime(2020,11,1)},'site_1')]
	pl = ParticleList()
	for float_pos_dict,filename in float_list:
		for start_day in [5]*3:
			float_pos_dict['time'] = float_pos_dict['time']+datetime.timedelta(days=start_day)
			uv = uv_class.load(float_pos_dict['time'],float_pos_dict['time']+datetime.timedelta(days=52))		
			uv.time.set_ref_date(float_pos_dict['time'])
			data,dimensions = uv.return_parcels_uv(float_pos_dict['time'],days_delta=52)
			prediction = UVPrediction(float_pos_dict,data,dimensions)
			prediction.create_prediction(ArgoVerticalMovement,days=50.)
			nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
			pl.append(nc)

	for k,timedelta in enumerate([datetime.timedelta(days=x) for x in range(50)]):
		XX,YY,ax = uv.plot()
		pl.plot_density(timedelta,[uv.lons,uv.lats],ax)
		plt.savefig(file_handler.out_file('pdf_movie/'+str(k)))
		plt.close()
	os.chdir(file_handler.out_file('pdf_movie/'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")
