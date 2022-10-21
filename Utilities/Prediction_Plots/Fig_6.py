from HyperNav.Utilities.Data.Instrument import return_HI21A55, return_HI21A54, return_HI21B55, return_HI21B54
from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
from HyperNav.Utilities.Compute.RunParcels import create_prediction,ParticleDataset
import matplotlib.pyplot as plt
from GeneralUtilities.Compute.constants import degree_dist, seconds_per_day
import math
import cartopy.crs as ccrs
import os
import datetime
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import itertools
import operator
import numpy as np
import geopy
from HyperNav.Utilities.__init__ import ROOT_DIR as PLOT_DIR

date_start = datetime.datetime(2021,6,1)
date_end = datetime.datetime(2021,7,1)
uv_class = HYCOMHawaii.load(date_start,date_end)
file_handler = FilePathHandler(ROOT_DIR,'HypernavHawaiiFieldDeployment')
plot_handler = FilePathHandler(PLOT_DIR,'Prediction_Figures')
plt.rcParams['font.size'] = '16'

dist_list = []
displacement_list = []
for Hypernav in [return_HI21A55(), return_HI21A54(), return_HI21B55(), return_HI21B54()]:
	for time_idx in range(len(Hypernav.time)-1):
		date = Hypernav.time[time_idx]
		loc = surface_loc = geopy.Point(Hypernav.lats[time_idx],Hypernav.lons[time_idx])
		float_pos_dict = Hypernav.return_float_pos_dict(date)
		uv,dimensions = uv_class.return_parcels_uv(date-datetime.timedelta(days=1),date+datetime.timedelta(days=4))
		if not os.path.isfile(file_handler.tmp_file(Hypernav.label+'_day_'+str(time_idx)+'.nc')):
			create_prediction(float_pos_dict,uv,dimensions,file_handler.tmp_file(Hypernav.label+'_day_'+str(time_idx)+'.nc'))
		nc = ParticleDataset(file_handler.tmp_file(Hypernav.label+'_day_'+str(time_idx)+'.nc'))
		try:
			for k,delta in enumerate([Hypernav.time[time_idx+x]-date for x in [1,2,3]]):
				projected_date = date+delta
				if projected_date>max(Hypernav.time):
					continue 
				lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
				surface_loc = geopy.Point(Hypernav.lats[time_idx+k+1],Hypernav.lons[time_idx+k+1])
				dist_list.append((geopy.distance.great_circle(geopy.Point(lat_center,lon_center),surface_loc).nm,k+1,Hypernav.label))
				displacement_list.append((geopy.distance.great_circle(loc,surface_loc).nm,delta.days,Hypernav.label))
		except IndexError:
			continue

fig = plt.figure(figsize=(10,10))
days_list = np.array([1,2,3])
for l,offset in [('A55',-0.1),('A54',-0.05),('B55',0.05),('B54',0.1)]:
	error_mean = []
	error_std = []
	for k in range(1,4):
		error_mean.append(np.mean([x for x,days,label in dist_list if (days==k)&(label==l)]))
		error_std.append(np.std([x for x,days,label in dist_list if (days==k)&(label==l)]))
	plt.scatter(days_list+offset,error_mean,label=l)
	plt.errorbar(days_list+offset,error_mean, yerr=error_std, fmt="o")
error_mean = []
error_std = []
for k in range(1,4):
	error_mean.append(np.mean([x for x,days,label in dist_list if (days==k)]))
	error_std.append(np.std([x for x,days,label in dist_list if (days==k)]))
plt.scatter(days_list,error_mean,color='black',label='All')
plt.errorbar(days_list,error_mean,color='black',yerr=error_std, fmt="o")
plt.xticks([1,2,3])
plt.xlabel('Days')
plt.ylabel('Prediction Error (nm)')
plt.ylim([0,13])
plt.legend(ncol=3,loc=2)
plt.savefig(plot_handler.out_file('Figure_'+str(6)),bbox_inches='tight')
plt.close()