import pickle
from pydap.client import open_url
import datetime
from GeneralUtilities.Compute.list import find_nearest, TimeList, LatList, LonList, DepthList
import numpy as np 

class Base(object):
	def __init__(self,u=None,v=None,lons=None,lats=None,time=None,depth=None,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.u = u
		self.v = v
		self.lons = LonList(lons)
		self.lats = LatList(lats)
		self.time = TimeList(time)
		self.depth = DepthList(depth)

		time_v,dummy,dummy,dummy = np.where(abs(self.v)>3)
		time_u,dummy,dummy,dummy = np.where(abs(self.u)>3)		
		remove = np.unique(time_v.tolist()+time_u.tolist()).tolist()
		time_mask = [x not in remove for x in range(len(self.time))]
		self.u = self.u[time_mask,:,:,:]
		self.v = self.v[time_mask,:,:,:]
		self.time = TimeList(np.array(self.time)[time_mask].tolist())

		assert isinstance(self.time,TimeList) 
		assert isinstance(self.depth,DepthList) 
		assert isinstance(self.lats,LatList) 
		assert isinstance(self.lons,LonList) 



	def return_u_v(self,time=None,depth=None):
		u_holder = self.u[self.time.find_nearest(time,idx=True),self.depth.find_nearest(depth,idx=True),:,:]
		v_holder = self.v[self.time.find_nearest(time,idx=True),self.depth.find_nearest(depth,idx=True),:,:]
		return (u_holder,v_holder)

	def return_monthly_mean(self,month,depth):
		mask = [x.month==month for x in self.time]
		u = self.u[mask,self.depth.find_nearest(depth,idx=True),:,:]
		v = self.v[mask,self.depth.find_nearest(depth,idx=True),:,:]
		return (np.nanmean(u,axis=0),np.nanmean(v,axis=0))

	@classmethod
	def make_filename(cls):
		return cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location)

	def save(self,filename=False):
		if not filename:
			filename = self.make_filename()
		with open(filename, 'wb') as pickle_file:
			pickle.dump(self,pickle_file)
		pickle_file.close()

	def vertical_shear(self,date,lat,lon):
		date_idx = self.time.find_nearest(date,idx=True)
		lat_idx = self.lats.find_nearest(lat,idx=True)
		lon_idx = self.lons.find_nearest(lon,idx=True)
		return (self.u[date_idx,:,lat_idx,lon_idx],self.v[date_idx,:,lat_idx,lon_idx])

	@classmethod
	def load(cls):
		with open(cls.make_filename(),'rb') as pickle_file:
			out_data = pickle.load(pickle_file)
		out_data = cls.new_from_old(out_data)

		return out_data

	@classmethod
	def new_from_old(cls,out):
		return cls(u=out.u,v=out.v,lons=out.lons,lats=out.lats,time=out.time,depth=out.depth)

	def plot(self,ax=False):
		return self.plot_class(self.lats,self.lons,ax).get_map()

	def return_parcels_uv(self,start_date):
		end_date = start_date+datetime.timedelta(days=5) 
		time_mask = [(x>start_date)&(x<end_date) for x in self.time]

		out_time = TimeList(np.array(self.time)[time_mask].tolist())
		out_u = self.u[time_mask,:,:,:]
		out_v = self.v[time_mask,:,:,:]
		out_w = np.zeros(out_u.shape)
		data = {'U':out_u,'V':out_v,'W':out_w}
		dimensions = {'time':out_time.seconds_since(),
		'depth':self.depth,
		'lat':self.lats,
		'lon':self.lons,}		
		return (data,dimensions)

	@classmethod
	def download_and_save(cls):
		dataset = open_url(cls.base_html+cls.ID)
		#open dataset using erdap server
		time_since = datetime.datetime.strptime(dataset['time'].attributes['time_origin'],'%d-%b-%Y %H:%M:%S')
		TimeList.set_ref_date(time_since)
		time = TimeList.time_list_from_seconds(dataset['time'][:])
		lats = LatList(dataset['latitude'][:])
		lons = dataset['longitude'][:].data
		lons[lons>180] = lons[lons>180]-360
		lons = LonList(lons)

		depth = -dataset['depth'][:].data
		depth = DepthList(depth)
		depth_idx = depth.find_nearest(-700,idx=True)
		depth = depth[:(depth_idx+1)]

		higher_lon_idx = lons.find_nearest(cls.urlon,idx=True)
		lower_lon_idx = lons.find_nearest(cls.lllon,idx=True)
		lons = lons[lower_lon_idx:higher_lon_idx]

		higher_lat_idx = lats.find_nearest(cls.urlat,idx=True)
		lower_lat_idx = lats.find_nearest(cls.lllat,idx=True)
		lats = lats[lower_lat_idx:higher_lat_idx]
		#define necessary variables from self describing netcdf 
		attribute_dict = dataset.attributes['NC_GLOBAL']
		time_end = datetime.datetime.strptime(attribute_dict['time_coverage_end'],'%Y-%m-%dT%H:%M:%SZ')


		time_idx_list = list(range(len(time))[::100])+[len(time)]
		u_list = []
		v_list = []
		k = 0
		while k < len(time_idx_list)-1:
			print(k)
			try:
				u_holder = dataset['water_u'][time_idx_list[k]:time_idx_list[k+1]
				,:(depth_idx+1)
				,lower_lat_idx:higher_lat_idx
				,lower_lon_idx:higher_lon_idx]
				v_holder = dataset['water_v'][time_idx_list[k]:time_idx_list[k+1]
				,:(depth_idx+1)
				,lower_lat_idx:higher_lat_idx
				,lower_lon_idx:higher_lon_idx]
				u_list.append(u_holder)
				v_list.append(v_holder)
				k +=1
			except:
				continue
		u = np.concatenate([np.array(x['water_u']) for x in u_list])
		v = np.concatenate([np.array(x['water_v']) for x in v_list])
		assert u.shape==v.shape
		assert u.shape[0] == len(time)
		assert u.shape[1] == len(depth)
		assert u.shape[2] == len(lats)
		assert u.shape[3] == len(lons)
		out = cls(u=u,v=v,lons=lons,lats=lats,time=time,depth=depth)
		out.save()