from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import os
import xroms
import numpy as np
import datetime
import matplotlib.pyplot as plt
from GeneralUtilities.Data.pickle_utilities import save

file_handler = FilePathHandler(ROOT_DIR,'WGOFS')
location='SoCal'
dataset_description = 'WGOFS'
folder_name = file_handler.tmp_file(dataset_description+'_'+location+'_historical_data')
hours_list = np.arange(0,120,3).tolist()
dataset_time = [datetime.datetime(2024,9,23)+datetime.timedelta(hours=x) for x in hours_list]

""" Have to do this because the packages for geospatial interpolation get confused in conda. Use environment xesmf_env"""

def return_dims():
	lats = np.linspace(32.5,33.7,200)
	lons = np.linspace(-117,-118,200)
	depths = np.linspace(0,-500,50)
	return (lats,lons,depths)


def return_uv_roms_dict(filename):
	outdict = {}
	ds = xroms.open_netcdf(filename)
	ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=False, include_Z0=True)
	lats,lons,depths = return_dims()
	for varname,varin in [('u',ds.u),('v',ds.v)]:
		varout = xroms.interpll(varin,lons,lats, which='grid')
		outdict[varname] = xroms.isoslice(varout.chunk(-1),depths,xgrid)
	return outdict

def process_data():
	u_list = []
	v_list = []
	time_list = []
	for time in dataset_time:
		print(time)
		filename = time.strftime('wcofs.t03z.%Y%m%d.fields.f{0:03d}.nc'.format(time.hour))
		k_filename = os.path.join(folder_name,filename)
		try:
			data_dict = return_uv_roms_dict(k_filename)
		except FileNotFoundError:
			time=time-datetime.timedelta(days=1)
			filename = time.strftime('wcofs.t03z.%Y%m%d.fields.f{0:03d}.nc'.format(24))
			k_filename = os.path.join(folder_name,filename)
			data_dict = return_uv_roms_dict(k_filename)
		u_list.append(np.array(data_dict['u'].data))
		v_list.append(np.array(data_dict['v'].data))
		time_list.append(time)

	v = np.stack(v_list)
	np.save(os.path.join(folder_name,'v'),v)
	u = np.stack(u_list)
	np.save(os.path.join(folder_name,'u'),u)
