from HyperNav.Utilities.Data.UVBase import Base
from GeneralUtilities.Plot.Cartopy.eulerian_plot import KonaCartopy
import numpy as np 
import datetime
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
file_handler = FilePathHandler(ROOT_DIR,'HYCOMBase')


class HYCOMBase(Base):
	dataset_description = 'HYCOM'
	hours_list = np.arange(0,25,3).tolist()
	base_html = 'https://www.ncei.noaa.gov/erddap/griddap/'
	file_handler = file_handler
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class HYCOMHawaii(HYCOMBase):
	plot_class = KonaCartopy
	location='Hawaii'
	urlat = 20.5
	lllat = 18.8
	lllon = -157
	urlon = -155.5
	ID = 'HYCOM_reg6_latest3d'
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.time.set_ref_date(datetime.datetime(1970,1,1))

		
# class HYCOMPuertoRico	
# 	120 W 50 W 
# 	ID = 'HYCOM_reg1_latest3d'


# class HYCOMTahiti
# 	150 W 120 W 
# 	ID = 'HYCOM_reg7_latest3d'


# class ReturnHYCOMUV(HYCOMDataOpenAndParse):
# 	def __init__(self,float_pos_dict,*args,days=5,**kwargs):
# 		super().__init__(float_pos_dict,*args,**kwargs)
# 		depth_idx = 35
# 		end_time_idx = self.time_idx+len(self.hours_list)*days
# 		U = self.dataset['water_u'][self.time_idx:end_time_idx
# 		,:depth_idx
# 		,self.lower_lat_idx:self.higher_lat_idx
# 		,self.lower_lon_idx:self.higher_lon_idx]
# 		V = self.dataset['water_v'][self.time_idx:end_time_idx
# 		,:depth_idx
# 		,self.lower_lat_idx:self.higher_lat_idx
# 		,self.lower_lon_idx:self.higher_lon_idx]
# 		W = np.zeros(U.data[0].shape)
# 		self.data = {'U':U.data[0],'V':V.data[0],'W':W}
		
# 		self.dimensions = {'time':self.time.seconds_since()[self.time_idx:end_time_idx],
# 		'depth':list(self.dataset['depth'][:depth_idx]),
# 		'lat':self.lats,
# 		'lon':self.lons,}

# 	def get_new_snapshot(self,float_pos_dict,days=5):
# 		class BlankUV():
# 			pass

# 		holderUV = BlankUV()
# 		TimeList.set_ref_date(self.time.ref_date)
# 		time = TimeList.time_list_from_seconds(self.dimensions['time'])
# 		start_date = float_pos_dict['datetime'].date()
# 		start_date = datetime.datetime.fromordinal(start_date.toordinal())
# 		end_date = start_date+datetime.timedelta(days=days)
# 		start_time_idx = time.closest_index(start_date)
# 		end_time_idx = time.closest_index(end_date)
# 		time = time.seconds_since()[start_time_idx:end_time_idx]
# 		U = self.data['U'][start_time_idx:end_time_idx,:,:,:]
# 		V = self.data['V'][start_time_idx:end_time_idx,:,:,:]
# 		W = self.data['W'][start_time_idx:end_time_idx,:,:,:]
# 		holderUV.data = {'U':U,'V':V,'W':W}
# 		holderUV.float_pos_dict = float_pos_dict
# 		holderUV.dimensions = {'time':time,
# 		'depth':self.dimensions['depth'],
# 		'lat':self.dimensions['lat'],
# 		'lon':self.dimensions['lon']}
# 		return holderUV

# class UVPrediction():

# 	def __init__(self,float_pos_dict,uv=False,*args,**kwargs):
# 		if not uv:
# 			uv = ReturnHYCOMUV(float_pos_dict,*args,**kwargs)
# 		self.float_pos_dict = float_pos_dict
# 		self.uv = uv

# 	def create_prediction(self,vert_move,days=3):
# 		fieldset = FieldSet.from_data(self.uv.data, self.uv.dimensions,transpose=False)
# 		fieldset.mindepth = self.uv.dimensions['depth'][0]
# 		K_bar = 0.000000000025
# 		fieldset.add_constant('Kh_meridional',K_bar)
# 		fieldset.add_constant('Kh_zonal',K_bar)
# 		testParticles = get_test_particles(fieldset,self.float_pos_dict,self.uv.dimensions['time'][0])
# 		kernels = vert_move + testParticles.Kernel(AdvectionRK4)
# 		dt = 15 #15 minute timestep
# 		output_file = testParticles.ParticleFile(name=file_handler.tmp_file('Uniform_out.nc'),
# 			outputdt=datetime.timedelta(minutes=dt))
# 		testParticles.execute(kernels,
# 							  runtime=datetime.timedelta(days=days),
# 							  dt=datetime.timedelta(minutes=dt),
# 							  output_file=output_file,)
# 		output_file.export()
# 		output_file.close()

# 	def calculate_prediction(self,depth_level,days=3.):
# 		predictions = []
# 		self.create_prediction(vert_move_dict[depth_level],days=days)
# 		nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
# 		nc['cycle_age'][0,:].data
# 		holder = nc['time'][0,:]
# 		assert ([x-holder[0] for x in holder][:10] == nc['cycle_age'][0,:].data[:10]).all()
# 		#time must be passing the same for the float
# 		for k,time in enumerate([datetime.timedelta(days=x) for x in np.arange(.2,days,.1)]):
# 			try: 
# 				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(time)
# 			except ValueError:
# 				continue
# 			date_string = (self.float_pos_dict['datetime']+time).isoformat()
# 			id_string = int(str(self.float_pos_dict['ID'])+'0'+str(depth_level)+'0'+str(k))
# 			dummy_dict = {"prediction_id":id_string,
# 			"datetime":date_string,
# 			"lat":float(lat_center),
# 			"lon":float(lon_center),
# 			"uncertainty":0,
# 			"model":'HYCOM'+'_'+str(depth_level)}
# 			predictions.append(dummy_dict)
# 		return predictions

# 	def upload_single_depth_prediction(self,depth_level):
# 		SiteAPI.delete_by_model('HYCOM'+'_'+str(depth_level),self.float_pos_dict['ID'])
# 		predictions = self.calculate_prediction(depth_level,days=1)
# 		SiteAPI.upload_prediction([x for x in predictions if x['model']=='HYCOM'+'_'+str(depth_level)],self.float_pos_dict['ID'])

# 	def upload_multi_depth_prediction(self):
# 		for depth_level in vert_move_dict.keys():
# 			self.upload_single_depth_prediction(depth_level)

# 	def plot_multi_depth_prediction(self):
# 		color_dict = {50:'red',100:'purple',200:'blue',300:'teal',
# 		400:'pink',500:'tan',600:'orange',700:'yellow'}
# 		self.create_prediction(vert_move_dict[50])
# 		nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
# 		XX,YY,ax = HypernavCartopy(nc,self.float_pos_dict,lon_grid=self.uv.lons,lat_grid=self.uv.lats,pad=-0.5).get_map()
# 		depth = Depth()
# 		XX1,YY1 = np.meshgrid(depth.x,depth.y)
# 		plt.contour(XX1,YY1,depth.z,[-1*self.float_pos_dict['park_pressure']],colors=('k',),linewidths=(4,),zorder=4,label='Drift Depth Contour')
# 		plt.contourf(XX1,YY1,np.ma.masked_greater(depth.z/1000.,0),zorder=3,cmap=plt.get_cmap('Greys'))
# 		plt.colorbar(label='Depth (km)')
# 		plt.scatter(self.float_pos_dict['lon'],self.float_pos_dict['lat'],marker='x',c='k',linewidth=6,s=250,zorder=6,label='Location')
# 		for k in range(particle_num):
# 			lats = nc['lat'][k,:]
# 			lons = nc['lon'][k,:]
# 			plt.plot(lons,lats,linewidth=2,zorder=10)
# 		plt.title('Float '+str(self.float_pos_dict['ID'])+' at '+datetime.datetime.now().isoformat())
# 		savefile =str(self.float_pos_dict['ID'])+'_Multidepth_'+str(self.float_pos_dict['profile'])		
# 		plt.savefig(file_handler.out_file(savefile))
# 		plt.close()	


# 	def bonus():
# 		for vert_move,drift_depth in zip([ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50],[700,600,500,400,300,200,100,50]):
# 			percent_aground,percent_lahaina_in_bounds,percent_kona_in_bounds,float_move = create_prediction(float_pos_dict,vert_move,drift_depth)
# 			plot_total_prediction(drift_depth)
# 			plot_snapshot_prediction(drift_depth)
# 			nc = ParticleDataset(file_handler.tmp_file('Uniform_out.nc'))
# 			percent_aground = nc.percentage_aground(depth_level = drift_depth)
# 			percent_lahaina_in_bounds = nc.within_bounds(lahaina_pos)
# 			percent_kona_in_bounds = nc.within_bounds(kona_pos)
# 			float_move = nc.dist_list(float_pos_dict)
# 			aground_list.append((percent_aground,drift_depth))
# 			lahaina_list.append((percent_lahaina_in_bounds,drift_depth))
# 			kona_list.append((percent_kona_in_bounds,drift_depth))
# 			float_move_list.append((float_move,drift_depth))

# 		plt.bar(depth,aground_percent,width=25)
# 		plt.ylabel('Grounding Percentage')
# 		plt.xlabel('Depth')
# 		plt.savefig(file_handler.out_file('grounding_percentage'))
# 		plt.close()

# 		def plot_position(pos_list):
# 			for percent_outside,depth in pos_list:
# 				plt.plot([1,2,3],percent_outside,label=(str(depth)+'m Depth'))
# 			plt.legend()
# 			plt.xlabel('Days')


# 		plt.figure()
# 		plot_position(lahaina_list)
# 		plt.ylabel('Percent of Floats Outside Operations Area')
# 		plt.savefig(file_handler.out_file('lahaina_percentage'))
# 		plt.close()

# 		plt.figure()
# 		plot_position(kona_list)
# 		plt.ylabel('Percent of Floats Outside Operations Area')
# 		plt.savefig(file_handler.out_file('kona_percentage'))
# 		plt.close()

# 		plt.figure()
# 		plot_position(float_move_list)
# 		plt.ylabel('Distance from Starting Point Floats Move (nm)')
# 		plt.savefig(file_handler.out_file('float_dist'))
# 		plt.close()
