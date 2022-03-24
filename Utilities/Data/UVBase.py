import pickle
from pydap.client import open_url
import datetime
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList,flat_list
from GeneralUtilities.Compute.constants import degree_dist
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
	time_method = UVTimeList.time_list_from_seconds
	def __init__(self,u=None,v=None,lons=None,lats=None,time=None,depth=None,units=None,*args,**kwargs):
		super().__init__(*args,**kwargs)
		assert (units == 'm/s')|(units=='meters/second')|(units=='m s-1')
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

	@classmethod
	def calculate_divergence(cls,u,v):
		time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,units = cls.get_dimensions()
		XX,YY = np.meshgrid(lons,lats)
		dx = np.gradient(XX)[1]*degree_dist*1000
		dy = np.gradient(YY)[0]*degree_dist*1000
		du_dx = np.gradient(u,axis=1)/dx
		dv_dy = np.gradient(v,axis=0)/dy
		return du_dx+dv_dy

	@classmethod
	def calculate_curl(cls,u,v):
		time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,units = cls.get_dimensions()
		XX,YY = np.meshgrid(lons,lats)
		dx = np.gradient(XX)[1]*degree_dist*1000
		dy = np.gradient(YY)[0]*degree_dist*1000
		du_dy = np.gradient(u,axis=0)/dy
		dv_dx = np.gradient(v,axis=1)/dx
		return dv_dx-du_dy

	def subsample_time_u_v(self,N):
		u = self.u[::N,:,:,:]
		v = self.v[::N,:,:,:]
		time = self.time[::N]
		return self.__class__(u=u,v=v,lons=self.lons,lats=self.lats,time=time,depth=self.depth,units='m/s')

	def subsample_space_u_v(self,lllon,urlon,lllat,urlat):
		lllon_index = self.lons.find_nearest(lllon,idx=True)
		urlon_index = self.lons.find_nearest(urlon,idx=True)
		lllat_index = self.lats.find_nearest(lllat,idx=True)
		urlat_index = self.lats.find_nearest(urlat,idx=True)
		u = self.u[:,:,lllat_index:urlat_index,lllon_index:urlon_index]
		v = self.v[:,:,lllat_index:urlat_index,lllon_index:urlon_index]
		lons = self.lons[lllon_index:urlon_index]
		lats = self.lats[lllat_index:urlat_index]
		return self.__class__(u=u,v=v,lons=lons,lats=lats,time=self.time,depth=self.depth,units='m/s')

	def subsample_depth(self,N,max_depth=None):
		u = self.u[:,::N,:,:]
		v = self.v[:,::N,:,:]
		depth = self.depth[::N]
		if max_depth:
			depth_idx = depth.find_nearest(max_depth,idx=True)
			u = u[:,:depth_idx,:,:]
			v = v[:,:depth_idx,:,:]
			depth = depth[:depth_idx]
		return self.__class__(u=u,v=v,lons=self.lons,lats=self.lats,time=self.time,depth=depth,units='m/s')



	def return_u_v(self,time=None,depth=None):
		u_holder = self.u[self.time.find_nearest(time,idx=True),self.depth.find_nearest(depth,idx=True),:,:]
		v_holder = self.v[self.time.find_nearest(time,idx=True),self.depth.find_nearest(depth,idx=True),:,:]
		return (u_holder,v_holder)

	def return_monthly_mean(self,month,depth):
		mask = [x.month==month for x in self.time]
		u = self.u[mask,self.depth.find_nearest(depth,idx=True),:,:]
		v = self.v[mask,self.depth.find_nearest(depth,idx=True),:,:]
		return (np.nanmean(u,axis=0),np.nanmean(v,axis=0))

	def return_monthly_std(self,month,depth):
		mask = [x.month==month for x in self.time]
		u = self.u[mask,self.depth.find_nearest(depth,idx=True),:,:]
		v = self.v[mask,self.depth.find_nearest(depth,idx=True),:,:]
		return (np.nanstd(u,axis=0),np.nanstd(v,axis=0))

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
		time,lats,lons,depth,lower_lon_idx,higher_lon_idx,lower_lat_idx,higher_lat_idx,units = cls.get_dimensions()
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
			time_list.append(cls.time_method(uv_dict['time']))
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
		out = cls(u=u,v=v,lons=lons,lats=lats,time=time,depth=depth,units=units)
		return out

	@classmethod
	def new_from_old(cls,out):
		return cls(u=out.u,v=out.v,lons=out.lons,lats=out.lats,time=out.time,depth=out.depth)

	def plot(self,ax=False):
		return self.PlotClass(self.lats,self.lons,ax).get_map()

	def return_parcels_uv(self,start_date,days_delta=5):
		def add_zeros(array):
			array[:,:,:3,:]=0
			array[:,:,-3:,:]=0
			array[:,:,:,:3]=0
			array[:,:,:,-3:]=0
			return array

		end_date = start_date+datetime.timedelta(days=days_delta) 
		time_mask = [(x>start_date)&(x<end_date) for x in self.time]

		out_time = TimeList(np.array(self.time)[time_mask].tolist())
		out_u = self.u[time_mask,:,:,:]
		out_u = add_zeros(out_u)
		out_v = self.v[time_mask,:,:,:]
		out_v = add_zeros(out_v)
		out_w = np.zeros(out_u.shape)
		data = {'U':out_u,'V':out_v,'W':out_w}
		dimensions = {'time':out_time.seconds_since(),
		'depth':[-x for x in self.depth],
		'lat':self.lats,
		'lon':self.lons,}		
		return (data,dimensions)