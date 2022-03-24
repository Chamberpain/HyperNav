from HyperNav.Data.__init__ import ROOT_DIR
import geopandas as gp
from HyperNav.Utilities.Data.Copernicus import Copernicus
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


class EEZ(object):
	def __init__(self):
		eez_df = gp.read_file(os.path.join(ROOT_DIR,'Intersect_EEZ_IHO_v4_2020/Intersect_EEZ_IHO_v4_2020.shp'))
		coord_list = [(self.data_class.urlon,self.data_class.urlat),(self.data_class.urlon,self.data_class.lllat),
		(self.data_class.lllon,self.data_class.lllat),(self.data_class.lllon,self.data_class.urlat),
		(self.data_class.urlon,self.data_class.urlat)]
		ocean_region = shapely.geometry.Polygon(coord_list)
		self.df = eez_df[~eez_df.intersection(ocean_region).is_empty]
		self.df['geometry'] = self.df.intersection(ocean_region)

	def plot(self,ax):
		transform = ccrs.PlateCarree()._as_mpl_transform(ax)
		for idx,row in self.df.iterrows():
			ax.add_geometries([row.geometry], crs=ccrs.PlateCarree(),facecolor='white',edgecolor='black',alpha=0.5,zorder=0)
			ax.annotate(row.TERRITORY1,xy=(row.geometry.centroid.x,row.geometry.centroid.y),fontsize=12,xycoords=transform, zorder=12,bbox=dict(boxstyle="round,pad=0.3",alpha=0.75))
		return ax

class CreteEEZ(EEZ):
	data_class = Copernicus


