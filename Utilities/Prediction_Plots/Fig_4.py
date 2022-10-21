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

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
uv_class.PlotClass.urcrnrlat=20
uv_class.PlotClass.llcrnrlat=19
uv_class.PlotClass.urcrnrlon=-155.8
uv_class.PlotClass.llcrnrlon=-156.8

X,Y,ax = uv_class.plot(ax=ax)


for Hypernav,color in [(return_HI21A55(),'blue'), (return_HI21A54(),'green'), (return_HI21B55(),'orange'), (return_HI21B54(),'red')]:
	loc_list = []
	predict_list = []
	for time_idx in range(len(Hypernav.time)-1):
		date = Hypernav.time[time_idx]
		loc = surface_loc = geopy.Point(Hypernav.lats[time_idx],Hypernav.lons[time_idx])
		float_pos_dict = Hypernav.return_float_pos_dict(date)
		uv,dimensions = uv_class.return_parcels_uv(date-datetime.timedelta(days=1),date+datetime.timedelta(days=4))
		if not os.path.isfile(file_handler.tmp_file(Hypernav.label+'_day_'+str(time_idx)+'.nc')):
			create_prediction(float_pos_dict,uv,dimensions,file_handler.tmp_file(Hypernav.label+'_day_'+str(time_idx)+'.nc'))
		nc = ParticleDataset(file_handler.tmp_file(Hypernav.label+'_day_'+str(time_idx)+'.nc'))
		try:
			delta = Hypernav.time[time_idx+1]-date
			projected_date = date+delta
			if projected_date>max(Hypernav.time):
				continue 
			lat_center,lon_center,lat_std,lon_std = nc.get_cloud_center(delta)
			loc_list.append(geopy.Point(Hypernav.lats[time_idx+1],Hypernav.lons[time_idx+1]))
			predict_list.append(geopy.Point(lat_center,lon_center))
		except IndexError:
			continue
	print(loc_list)
	print(predict_list)
	loc_lat,loc_lon = zip(*[(x.latitude,x.longitude) for x in loc_list])
	ax.plot(loc_lon,loc_lat,color=color,label=Hypernav.label)
	predict_lat,predict_lon = zip(*[(x.latitude,x.longitude) for x in predict_list])
	ax.plot(predict_lon,predict_lat,color=color,linestyle='dashed')
ax.legend()
plt.savefig(plot_handler.out_file('Figure_'+str(4)),bbox_inches='tight')