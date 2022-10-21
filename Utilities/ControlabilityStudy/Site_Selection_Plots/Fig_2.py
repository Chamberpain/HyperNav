from HyperNav.Utilities.ControlabilityStudy.ControlBase import ControlHawaii
import geopy
import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

deployed_point = geopy.Point(19.5,-156.4)
start_time = datetime.datetime(2018,1,3)
start_time_list,start_point_list,drift_depth_list = ControlHawaii().maintain_location(deployed_point,start_time,0)
lat_list = [x.latitude for x in start_point_list]
lon_list = [x.longitude for x in start_point_list]

uv_class = ControlHawaii.uv_class.load(start_time,start_time+datetime.timedelta(days=2))
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
uv_class.PlotClass.urcrnrlat=max(lat_list)+0.2
uv_class.PlotClass.llcrnrlat=min(lat_list)-0.2
uv_class.PlotClass.urcrnrlon=max(lon_list)+0.2
uv_class.PlotClass.llcrnrlon=min(lon_list)-0.2
X,Y,ax = uv_class.plot(ax=ax)
ax.plot(lon_list,lat_list,color='black')
pbc = ax.scatter(lon_list,lat_list,c=drift_depth_list,zorder=20)
ax.scatter(lon_list[0],lat_list[0],c='black',marker='*',s=500,zorder=21)
ax.scatter(lon_list[0],lat_list[0],c='pink',marker='*',s=350,zorder=21)
ax.scatter(lon_list[-1],lat_list[-1],c='black',marker='s',s=400,zorder=21)
ax.scatter(lon_list[-1],lat_list[-1],c='pink',marker='s',s=300,zorder=21)

plt.colorbar(mappable=pbc,label='Drift Depth')