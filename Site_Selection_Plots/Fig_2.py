import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from HyperNav.Utilities.Data.CopernicusMed import CreteCopernicus
from HyperNav.Utilities.Data.HYCOM import HYCOMSouthernCalifornia,HYCOMHawaii,HYCOMAlaska,HYCOMPuertoRico
from HyperNav.Utilities.Data.CopernicusGlobal import TahitiCopernicus
from HyperNav.Utilities.Data.PACIOOS import KonaPACIOOS
import geopy
from GeneralUtilities.Plot.Cartopy.eulerian_plot import BaseCartopy
from GeneralUtilities.Compute.list import GeoList 


class RobinsonCartopy(BaseCartopy):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,ax = plt.axes(projection=ccrs.Robinson()),**kwargs)      
		print('I am plotting global region')
		self.finish_map()

# fig = plt.figure(figsize=(18,12))
# ax = fig.add_subplot(1,1,1)

def make_plot():
	XX, YY, ax = RobinsonCartopy().get_map()
	ax.set_global()
	locs = GeoList([geopy.Point(19.5,-156.4),geopy.Point(33.7,-119.6),geopy.Point(36.7,-122.2),geopy.Point(17.8,-66.7,0),geopy.Point(35.75,25.0,0),geopy.Point(-17.8,-149.75,0),geopy.Point(32,-64.5,0)])
	lats,lons = locs.lats_lons()
	ax.scatter(lons,lats,transform=ccrs.PlateCarree(),s=80,marker='*',c='r',zorder=1000)
	transform = ccrs.PlateCarree()._as_mpl_transform(ax)
	for geo in [TahitiCopernicus,CreteCopernicus,HYCOMSouthernCalifornia,HYCOMHawaii,HYCOMAlaska,HYCOMPuertoRico,KonaPACIOOS]:
		for shape in geo.get_dataset_shape():
			ax.add_geometries([shape], crs=ccrs.PlateCarree(), facecolor=geo.facecolor,edgecolor='black',alpha=0.25)
			ax.annotate(geo.dataset_description,xy=(shape.centroid.x,shape.centroid.y+5),fontsize=12,xycoords=transform, zorder=12,bbox=dict(boxstyle="round,pad=0.3", fc=geo.facecolor,alpha=0.75))
