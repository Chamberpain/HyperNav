import pickle
from pydap.client import open_url
import datetime
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList,flat_list
import numpy as np 
import os

class UVTimeList(TimeList):
	def return_time_list(self):
		return list(range(len(self))[::100])+[len(self)-1] #-1 because of pythons crazy list indexing

	def get_file_indexes(self,start_time,end_time):
		assert start_time<end_time
		key_list = TimeList([self[x] for x in self.return_time_list()])
		start_idx = key_list.find_nearest(start_time,idx = True)
		end_idx = key_list.find_nearest(end_time,idx = True)
		return np.arange(start_idx-1,end_idx+2).tolist() # +2 because we want to have +1 


class Base(object):
	def __init__(self,u=None,v=None,lons=None,lats=None,time=None,depth=None,units=None,*args,**kwargs):
		super().__init__(*args,**kwargs)
		assert units == 'm/s'
		self.u = u
		self.v = v
		self.lons = LonList(lons)
		self.lats = LatList(lats)
		self.time = TimeList(time)
		self.depth = DepthList(depth)

		self.u = np.ma.masked_array(self.u,mask=abs(self.u)>3)
		self.v = np.ma.masked_array(self.v,mask=abs(self.v)>3)


		# time_v,dummy,dummy,dummy = np.where(abs(self.v)>3)
		# time_u,dummy,dummy,dummy = np.where(abs(self.u)>3)		
		# remove = np.unique(time_v.tolist()+time_u.tolist()).tolist()
		# time_mask = [x not in remove for x in range(len(self.time))]
		# self.u = self.u[time_mask,:,:,:]
		# self.v = self.v[time_mask,:,:,:]
		# self.time = TimeList(np.array(self.time)[time_mask].tolist())

#make sure the variable axis are the proper class
		assert isinstance(self.time,TimeList) 
		assert isinstance(self.depth,DepthList) 
		assert isinstance(self.lats,LatList) 
		assert isinstance(self.lons,LonList) 

#make sure all the dimensions are consistent
		assert self.u.shape==self.v.shape
		assert len(self.time)==self.u.shape[0]
		assert len(self.depth)==self.u.shape[1]
		assert len(self.lats)==self.u.shape[2]
		assert len(self.lons)==self.u.shape[3]

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
	def load(cls,date_start,date_end):
		time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx = cls.get_dimensions()
		time_idx_list = time.get_file_indexes(date_start,date_end)
		u_list = []
		v_list = []
		time_list = []
		for k in time_idx_list[:-1]:
			print(k)
			k_filename = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'+str(k))
			try:
				with open(k_filename, 'rb') as f:
					uv_dict = pickle.load(f)
			except FileNotFoundError:
				print(k_filename+' not found')
				continue
			u_list.append(uv_dict['u'])
			v_list.append(uv_dict['v'])
			time_list.append(time.time_list_from_seconds(uv_dict['time']))
		u = np.concatenate([x for x in u_list])
		v = np.concatenate([x for x in v_list])
		time = flat_list(time_list)
		print(u.shape)
		print(len(time))
		assert u.shape==v.shape
		assert u.shape[0] == len(time)
		assert u.shape[1] == len(depth)
		assert u.shape[2] == len(lats)
		assert u.shape[3] == len(lons)
		out = cls(u=u,v=v,lons=lons,lats=lats,time=time,depth=depth)
		return out

	@classmethod
	def new_from_old(cls,out):
		return cls(u=out.u,v=out.v,lons=out.lons,lats=out.lats,time=out.time,depth=out.depth)

	def plot(self,ax=False):
		return self.PlotClass(self.lats,self.lons,ax).get_map()

	def return_parcels_uv(self,start_date,days_delta=5):
		end_date = start_date+datetime.timedelta(days=days_delta) 
		time_mask = [(x>start_date)&(x<end_date) for x in self.time]

		out_time = TimeList(np.array(self.time)[time_mask].tolist())
		out_u = self.u[time_mask,:,:,:]
		out_v = self.v[time_mask,:,:,:]
		out_w = np.zeros(out_u.shape)
		data = {'U':out_u,'V':out_v,'W':out_w}
		dimensions = {'time':out_time.seconds_since(),
		'depth':[-x for x in self.depth],
		'lat':self.lats,
		'lon':self.lons,}		
		return (data,dimensions)

	@classmethod
	def get_dimensions(cls):
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
		depth_idx = depth.find_nearest(-700,idx=True)
		depth = depth[:(depth_idx+1)]
		higher_lon_idx = lons.find_nearest(cls.urlon,idx=True)
		lower_lon_idx = lons.find_nearest(cls.lllon,idx=True)
		lons = lons[lower_lon_idx:higher_lon_idx]
		higher_lat_idx = lats.find_nearest(cls.urlat,idx=True)
		lower_lat_idx = lats.find_nearest(cls.lllat,idx=True)
		lats = lats[lower_lat_idx:higher_lat_idx]
		return (time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx)


	@classmethod
	def download_and_save(cls):
		dataset = open_url(cls.base_html+cls.ID)
		#open dataset using erdap server
		time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx = cls.get_dimensions()
		#define necessary variables from self describing netcdf 
		attribute_dict = dataset.attributes['NC_GLOBAL']

		idx_list = time.return_time_list()
		k = 0
		while k < len(idx_list)-1:
			print(k)
			k_filename = cls.file_handler.tmp_file(cls.dataset_description+'_'+cls.location+'_data/'+str(k))
			if os.path.isfile(k_filename):
				k +=1
				continue
			try:
				u_holder = dataset['water_u'][idx_list[k]:idx_list[k+1]
				,:(len(depth))
				,lower_lat_idx:higher_lat_idx
				,lower_lon_idx:higher_lon_idx]
				v_holder = dataset['water_v'][idx_list[k]:idx_list[k+1]
				,:(len(depth))
				,lower_lat_idx:higher_lat_idx
				,lower_lon_idx:higher_lon_idx]
				with open(k_filename, 'wb') as f:
					pickle.dump({'u':u_holder['water_u'].data,'v':v_holder['water_v'].data, 'time':u_holder['time'].data},f)
				k +=1
			except:
				print('Index ',k,' encountered an error and did not save. Trying again')
				continue