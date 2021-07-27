import numpy as np
import fnmatch
import os
from netCDF4 import Dataset 
import datetime
from GeneralUtilities.Compute.list import flat_list

raw_base_file = '/Users/pchamberlain/Data/Raw/PACIOOS/february/'
processed_base_file = '/Users/pchamberlain/Data/Processed/PACIOOS/'



def load_files():
	file_list = ['depth','lat','lon','time','u_array','v_array']
	return_list = []
	for file_name in file_list:
		load_file = np.load(processed_base_file+file_name+'.npy',allow_pickle=True)
		if file_name in ['u_array','v_array']:
			load_file = np.ma.masked_greater(load_file,999)
		return_list.append(load_file)
	return return_list

def time_parse(net_cdf_time):
	start_time = datetime.datetime.strptime(str(net_cdf_time.time_origin),'%d-%b-%Y %H:%M:%S')
	return [start_time + datetime.timedelta(seconds = _) for _ in net_cdf_time[:]]

def save_files():
	time_list = []
	u_list = []
	v_list = []

	filenames = os.listdir(raw_base_file)
	filenames = fnmatch.filter(filenames,'roms_hiig_reanalysis_*.nc')
	for i,filename in enumerate(filenames):
		nc_fid = Dataset(raw_base_file+filename)
		time_list.append(time_parse(nc_fid['time']))
		if i==0:
			lat = nc_fid['latitude'][:].data
			lon = nc_fid['longitude'][:].data
			depth = nc_fid['depth'][:].data
		u_list.append(nc_fid['u'])
		v_list.append(nc_fid['v'])


	Y = [_[0] for _ in time_list]
	u_array = np.vstack([x[:] for _,x in sorted(zip(Y,u_list))])
	v_array = np.vstack([x[:] for _,x in sorted(zip(Y,v_list))])
	time = np.array(sorted(flat_list(time_list)))

	np.save(processed_base_file+'depth',depth)
	np.save(processed_base_file+'lat',lat)
	np.save(processed_base_file+'lon',lon)
	np.save(processed_base_file+'time',time)
	np.save(processed_base_file+'u_array',u_array.data)
	np.save(processed_base_file+'v_array',v_array.data)

