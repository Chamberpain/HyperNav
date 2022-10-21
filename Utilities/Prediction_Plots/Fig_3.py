from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
from HyperNav.Utilities.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
import numpy as np
import os
import matplotlib.patheffects as path_effects
import math
plt.rcParams['font.size'] = '16'

file_handler = FilePathHandler(ROOT_DIR,'Prediction_Figures')


lat = 19.4
lon = -156.25
date_start = datetime.datetime(2021,8,1)
date_end = datetime.datetime(2021,9,1)
uv_class = HYCOMHawaii.load(date_start,date_end)
lat_idx = uv_class.lats.find_nearest(lat,idx=True)
lon_idx = uv_class.lons.find_nearest(lon,idx=True)
u_profile = uv_class.u[:,:,lat_idx,lon_idx]
v_profile = uv_class.v[:,:,lat_idx,lon_idx]
depths = uv_class.depths

fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(1,2,1)

for t in uv_class.time:
	u,v = uv_class.vertical_shear(t,lat,lon)
	ax1.plot(u,depths,c='blue',alpha=0.35,linewidth=0.4)
	ax1.plot(v,depths,c='orange',alpha=0.35,linewidth=0.4)

ax1.plot(u_profile.mean(axis=0),depths,c='blue',alpha=1,linewidth=6,label='u',
path_effects=[path_effects.withStroke(linewidth=8,foreground="k")])
ax1.plot(v_profile.mean(axis=0),depths,c='orange',alpha=1,linewidth=6,label='v',
path_effects=[path_effects.withStroke(linewidth=8,foreground="k")])
ax1.set_ylim([-700,0])
ax1.set_xlim([-0.65,0.65])
ax1.set_xlabel('Current Speed ($m\ s^{-1}$)')
ax1.set_ylabel('Depth (m)')
ax1.legend()
ax1.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

ax2 = fig.add_subplot(2,2,2, polar=True)
ax2.set_rlim([0,0.4])
plt.locator_params(axis='x', nbins=3)
for u_uv_class,v_uv_class in zip(u_profile.mean(axis=0),v_profile.mean(axis=0)):
	theta = math.atan2(v_uv_class,u_uv_class)
	r = np.sqrt(u_uv_class**2+v_uv_class**2)
	ax2.arrow(theta,(r*0.5),0,r,alpha = 0.5, width = 0.06,
         edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)
ax2.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
ax3 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
XX,YY,ax3 = uv_class.plot(ax=ax3)
ax3.scatter(lon,lat,s=100,zorder=10)
u = uv_class.u[:,0,:,:].mean(axis=0)
v = uv_class.v[:,0,:,:].mean(axis=0)
ax3.quiver(XX,YY,u,v,scale=7)
ax3.annotate('c', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
plt.savefig(file_handler.out_file('Figure_'+str(3)),bbox_inches='tight')
