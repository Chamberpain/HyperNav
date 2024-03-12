import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from HyperNav.Utilities.Data.CopernicusMed import CopernicusMed
from HyperNav.Utilities.Data.HYCOM import HYCOMSouthernCalifornia,HYCOMHawaii,HYCOMAlaska,HYCOMPuertoRico

from GeneralUtilities.Plot.Cartopy.eulerian_plot import BaseCartopy

class RobinsonCartopy(BaseCartopy):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,ax = plt.axes(projection=ccrs.Robinson()),**kwargs)      
		print('I am plotting global region')
		self.finish_map()

# fig = plt.figure(figsize=(18,12))
# ax = fig.add_subplot(1,1,1)

def make_plot():
	XX, YY, ax = RobinsonCartopy().get_map()
	transform = ccrs.PlateCarree()._as_mpl_transform(ax)
	for geo in [CopernicusMed,HYCOMSouthernCalifornia,HYCOMHawaii,HYCOMAlaska,HYCOMPuertoRico]:
		geo = geo()
		for shape in geo.ocean_shape:
			ax.add_geometries([shape], crs=ccrs.PlateCarree(), facecolor=geo.facecolor,edgecolor='black',alpha=0.5)
			ax.annotate(geo.facename,xy=(shape.centroid.x,shape.centroid.y),fontsize=12,xycoords=transform, zorder=12,bbox=dict(boxstyle="round,pad=0.3", fc=geo.facecolor,alpha=0.75))