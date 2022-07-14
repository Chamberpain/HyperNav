import pickle
from pydap.client import open_url
import datetime
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList,flat_list
from GeneralUtilities.Compute.constants import degree_dist
import numpy as np 
import os
import shapely
from abc import ABC,abstractmethod

class UVTimeList(TimeList):
	
	def return_time_list(self):
		return list(range(len(self))[::100])+[len(self)-1] #-1 because of pythons crazy list indexing

	def get_file_indexes(self,start_time,end_time):
		assert start_time<end_time
		assert start_time>min(self)
		assert end_time<max(self)
		key_list = TimeList([self[x] for x in self.return_time_list()])
		start_idx = key_list.find_nearest(start_time,idx = True)
		end_idx = key_list.find_nearest(end_time,idx = True)
		return np.arange(start_idx-1,end_idx+2).tolist() # +2 because we want to have +1 


class Base(ABC):
	time_method = UVTimeList.time_list_from_seconds
	scale_factor = 1

	def __init__(self,u=None,v=None,time=None,*args,**kwargs):
		assert (self.units == 'm/s')|(self.units=='meters/second')|(self.units=='m s-1')
		self.time = TimeList(time)
		self.u = u
		self.v = v
		self.u = np.ma.masked_array(self.u,mask=abs(self.u)>3)
		self.v = np.ma.masked_array(self.v,mask=abs(self.v)>3)

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
	@abstractmethod
	def get_dataset(cls):
		pass


	@classmethod
	@abstractmethod
	def get_dimensions(cls):
		pass

	@classmethod
	@abstractmethod
	def download_and_save(cls):
		pass

	@classmethod
	def calculate_divergence(cls,u,v):
		XX,YY = np.meshgrid(cls.lons,cls.lats)
		dx = np.gradient(XX)[1]*degree_dist*1000
		dy = np.gradient(YY)[0]*degree_dist*1000
		du_dx = np.gradient(u,axis=1)/dx
		dv_dy = np.gradient(v,axis=0)/dy
		return du_dx+dv_dy

	@classmethod
	def calculate_curl(cls,u,v):
		XX,YY = np.meshgrid(cls.lons,cls.lats)
		dx = np.gradient(XX)[1]*degree_dist*1000
		dy = np.gradient(YY)[0]*degree_dist*1000
		du_dy = np.gradient(u,axis=0)/dy
		dv_dx = np.gradient(v,axis=1)/dx
		return dv_dx-du_dy

	def subsample_time_u_v(self,N):
		u = self.u[::N,:,:,:]
		v = self.v[::N,:,:,:]
		time = self.time[::N]
		return self.__class__(u=u,v=v,time=time)

	def subsample_space_u_v(self,lllon,urlon,lllat,urlat):
		lllon_index = self.lons.find_nearest(lllon,idx=True)
		urlon_index = self.lons.find_nearest(urlon,idx=True)
		lllat_index = self.lats.find_nearest(lllat,idx=True)
		urlat_index = self.lats.find_nearest(urlat,idx=True)
		u = self.u[:,:,lllat_index:urlat_index,lllon_index:urlon_index]
		v = self.v[:,:,lllat_index:urlat_index,lllon_index:urlon_index]
		self.__class__.lons = self.lons[lllon_index:urlon_index]
		self.__class__.lats = self.lats[lllat_index:urlat_index]
		return self.__class__(u=u,v=v,time=self.time)

	def subsample_depth(self,N,max_depth=None):
		u = self.u[:,::N,:,:]
		v = self.v[:,::N,:,:]
		depth = self.depth[::N]
		if max_depth:
			depth_idx = depth.find_nearest(max_depth,idx=True)
			u = u[:,:depth_idx,:,:]
			v = v[:,:depth_idx,:,:]
			depth = depth[:depth_idx]
		self.__class__.depth=depth
		return self.__class__(u=u,v=v,time=self.time)

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
	def get_drifter_profs(cls,ReadClass):
		float_names = ReadClass.get_floats_in_box(cls.ocean_shape)
		float_list = [ReadClass.all_dict[x] for x in float_names]
		return float_list

	@classmethod
	def load(cls,date_start,date_end):
		time_idx_list = cls.dataset_time.get_file_indexes(date_start,date_end)
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
		assert u.shape[1] == len(cls.depth)
		assert u.shape[2] == len(cls.lats)
		assert u.shape[3] == len(cls.lons)
		out = cls(u=u*cls.scale_factor,v=v*cls.scale_factor,time=time)
		return out

	@classmethod
	def new_from_old(cls,out):
		return cls(u=out.u,v=out.v,lons=out.lons,lats=out.lats,time=out.time,depth=out.depth)

	def plot(self,ax=False):
		return self.PlotClass(self.lats,self.lons,ax).get_map()

	def return_parcels_uv(self,start_date,end_date):
		def add_zeros(array):
			array[:,:,:3,:]=0
			array[:,:,-3:,:]=0
			array[:,:,:,:3]=0
			array[:,:,:,-3:]=0
			return array

		end_date += self.time_step
		start_date -= self.time_step

		time_mask = [(x>=start_date)&(x<=end_date) for x in self.time]

		out_time = TimeList(np.array(self.time)[time_mask].tolist())
		out_u = self.u[time_mask,:,:,:]
		out_u = add_zeros(out_u)
		out_v = self.v[time_mask,:,:,:]
		out_v = add_zeros(out_v)
		out_w = np.zeros(out_u.shape)
		data = {'U':out_u,'V':out_v,'W':out_w}
		dimensions = {'time':[x.timestamp() for x in out_time],
		'depth':[-x for x in self.depth],
		'lat':self.lats,
		'lon':self.lons,}		
		return (data,dimensions)