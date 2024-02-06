from HyperNav.Utilities.__init__ import ROOT_DIR
import h5py
from GeneralUtilities.Data.Filepath.instance import get_data_folder
from GeneralUtilities.Compute.list import LatList,LonList,TimeList
import os
DATA_DIR_FROUIN = os.path.join(get_data_folder(),'Processed/HDF5/svc')
DATA_DIR_ZIBORDI = os.path.join(get_data_folder(),'Processed/HDF5/zibordi')
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from GeneralUtilities.Plot.Cartopy.regional_plot import BermudaCartopy,CreteCartopy,KonaCartopy,PuertoRicoCartopy, TahitiCartopy


class SVCBase():
	def __init__(self):
		data_frouin = h5py.File(os.path.join(DATA_DIR_FROUIN,self.file))
		data_zibordi = h5py.File(os.path.join(DATA_DIR_ZIBORDI,self.file))

		self.percent_svc_frouin = data_frouin['SVC criteria probability']
		self.percent_clear_frouin = data_frouin['clear sky probability']

		self.percent_svc_zibordi = data_zibordi['SVC criteria probability']
		self.percent_clear_zibordi = data_zibordi['clear sky probability']

		self.lat = LatList(data_frouin['lat'][:].tolist())
		self.lon = LonList(data_frouin['lon'][:].tolist())

	def return_data(self,field,lat,lon,month_idx):
		lat_idx = self.lat.find_nearest(lat,idx=True)
		lon_idx = self.lon.find_nearest(lon,idx=True)
		return field[(month_idx-1),lat_idx,lon_idx]

	def return_svc_matchup(self,lat,lon,month_idx):
		return np.random.uniform(0,1)<self.return_data(self.percent_svc_frouin,lat,lon,month_idx)

	def return_clearsky_matchup(self,lat,lon,month_idx):
		return np.random.uniform(0,1)<self.return_data(self.percent_clear_frouin,lat,lon,month_idx)

	def return_svc_matchup_zibordi(self,lat,lon,month_idx):
		return np.random.uniform(0,1)<self.return_data(self.percent_svc_zibordi,lat,lon,month_idx)

	def return_clearsky_matchup_zibordi(self,lat,lon,month_idx):
		return np.random.uniform(0,1)<self.return_data(self.percent_clear_zibordi,lat,lon,month_idx)

	def plot_matchups(self,month_idx):
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
		XX,YY,ax = self.PlotClass(self.lat[::-1],self.lon,ax).get_map()
		cf = ax.pcolor(XX,YY,self.percent_svc[month_idx-1,:,:]*100)
		fig.colorbar(cf,label='Chance of SVC Matchup (%)')
		plt.show()

class SVCBermuda(SVCBase):
	file = 'Bermuda.h5'
	PlotClass = BermudaCartopy

class SVCSoCal(SVCBase):
	file = 'California_bight.h5'

class SVCCrete(SVCBase):
	file = 'Crete.h5'
	PlotClass = CreteCartopy

class SVCHawaii(SVCBase):
	file = 'Hawaii.h5'
	PlotClass = KonaCartopy

class SVCMonterey(SVCBase):
	file = 'Monterey.h5'

class SVCPuertoRico(SVCBase):
	file = 'PuertoRico.h5'
	PlotClass = PuertoRicoCartopy

class SVCTahiti(SVCBase):
	file = 'Tahiti.h5'
	PlotClass = TahitiCartopy
