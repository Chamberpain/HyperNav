from HyperNav.Utilities.Data.Instrument import FloatBase
from HyperNav.Utilities.Data.HYCOM import HYCOMPuertoRico,HYCOMMonterey,HYCOMHawaii
from HyperNav.Utilities.Data.PACIOOS import KonaPACIOOS
from HyperNav.Utilities.Data.CopernicusGlobal import PuertoRicoCopernicus,HawaiiCopernicus, MontereyCopernicus
from GeneralUtilities.Compute.list import TimeList, LatList, LonList,DepthList
from GeneralUtilities.Data.Filepath.instance import get_data_folder 
import pandas as pd
import os
import geopy
import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from easy_mpl import taylor_plot
from easy_mpl.utils import version_info
import shapely.geometry

UVPR = HYCOMPuertoRico.load(HYCOMPuertoRico.dataset_time[1],HYCOMPuertoRico.dataset_time[-2],interpolate=False)
UVM = HYCOMMonterey.load(HYCOMMonterey.dataset_time[1],HYCOMMonterey.dataset_time[-2],interpolate=False)
UVH = HYCOMHawaii.load(HYCOMHawaii.dataset_time[1],HYCOMHawaii.dataset_time[-2],interpolate=False)
UVHPACIOOS = KonaPACIOOS.load(KonaPACIOOS.dataset_time[1],KonaPACIOOS.dataset_time[-2],interpolate=False)
UVPRCopernicus = PuertoRicoCopernicus.load(PuertoRicoCopernicus.dataset_time[1],PuertoRicoCopernicus.dataset_time[-2],interpolate=False)
UVMCopernicus = MontereyCopernicus.load(MontereyCopernicus.dataset_time[1],MontereyCopernicus.dataset_time[-2],interpolate=False)
UVHCopernicus = HawaiiCopernicus.load(HawaiiCopernicus.dataset_time[1],HawaiiCopernicus.dataset_time[-2],interpolate=False)

class AOMLFloat(FloatBase):
	def __init__(self,df,*args,**kwargs):
		self.time = TimeList(df.time)
		self.lats = LatList(df.latitude)
		self.lons = LonList(df.longitude)
		self.ve = df.ve
		self.vn = df.vn
		if len(df.ve)>4:
			self.persistence_ve = pd.concat([self.ve[4:],pd.Series([np.nan]*4)])
			self.persistence_vn = pd.concat([self.vn[4:],pd.Series([np.nan]*4)])
		else:
			self.persistence_ve = pd.Series([np.nan]*len(self.ve))
			self.persistence_vn = pd.Series([np.nan]*len(self.ve))		
		assert len(self.ve)==len(self.persistence_ve)
		self.model_ve = []
		self.model_vn = []
		for time,lat,lon in zip(self.time,self.lats,self.lons):
			point = geopy.Point(lat,lon)
			time = datetime.datetime(time.year,time.month,time.day,time.hour)
			if not self.shape.contains(shapely.geometry.Point(lon,lat)):
				self.model_ve.append(np.nan)
				self.model_vn.append(np.nan)
				continue				
			try:
				u_holder,v_holder = self.UV.return_u_v(time,-10,point)
			except AssertionError:
				self.model_ve.append(np.nan)
				self.model_vn.append(np.nan)
				continue
			self.model_ve.append(u_holder)
			self.model_vn.append(v_holder)
		assert len(self.ve)==len(self.model_ve)
		assert len(self.vn)==len(self.model_vn)
		super().__init__(*args,**kwargs)

class HawaiiHYCOMAOMLFloat(AOMLFloat):
	UV = UVH
	shape = shapely.geometry.Polygon([(UV.lllon,UV.urlat),(UV.urlon,UV.urlat),(UV.urlon,UV.lllat),(UV.lllon,UV.lllat)])

class HawaiiPACIOOSAOMLFloat(AOMLFloat):
	UV = UVHPACIOOS
	shape = shapely.geometry.Polygon([(UV.lllon,UV.urlat),(UV.urlon,UV.urlat),(UV.urlon,UV.lllat),(UV.lllon,UV.lllat)])

class PuertoRicoHYCOMAOMLFloat(AOMLFloat):
	UV = UVPR
	shape = shapely.geometry.Polygon([(UV.lllon,UV.urlat),(UV.urlon,UV.urlat),(UV.urlon,UV.lllat),(UV.lllon,UV.lllat)])

class MontereyHYCOMAOMLFloat(AOMLFloat):
	UV = UVM
	shape = shapely.geometry.Polygon([(UV.lllon,UV.urlat),(UV.urlon,UV.urlat),(UV.urlon,UV.lllat),(UV.lllon,UV.lllat)])

class PuertoRicoCopernicusAOMLFloat(AOMLFloat):
	UV = UVPRCopernicus
	shape = shapely.geometry.Polygon([(UV.lllon,UV.urlat),(UV.urlon,UV.urlat),(UV.urlon,UV.lllat),(UV.lllon,UV.lllat)])

class MontereyCopernicusAOMLFloat(AOMLFloat):
	UV = UVMCopernicus
	shape = shapely.geometry.Polygon([(UV.lllon,UV.urlat),(UV.urlon,UV.urlat),(UV.urlon,UV.lllat),(UV.lllon,UV.lllat)])

class HawaiiCopernicusAOMLFloat(AOMLFloat):
	UV = UVHCopernicus
	shape = shapely.geometry.Polygon([(UV.lllon,UV.urlat),(UV.urlon,UV.urlat),(UV.urlon,UV.lllat),(UV.lllon,UV.lllat)])


class AOMLFloatsBase():
	def __init__(self):
		self.float_list = []
		df = pd.read_csv(self.path)[1:]
		df.time = pd.to_datetime(df.time)
		df.latitude = df.latitude.astype(float)
		df.longitude = df.longitude.astype(float)
		df.ve = df.ve.astype(float)
		df.vn = df.vn.astype(float)
		df.vn = df.vn.astype(float)
		df = df[df.vn!=-999999.0]
		for ID in df.ID.unique():
			print(ID)
			dummy_df = df[df.ID==ID]
			self.float_list.append(self.FloatClass(dummy_df))

	def plot_model_data_misfit():
		x_error_list = []
		y_error_list = []
		for dummy in self.float_list:
			x_error_list+=(np.array(dummy.ve)-np.array(dummy.model_ve)).tolist()
			y_error_list+=(np.array(dummy.vn)-np.array(dummy.model_vn)).tolist()
		print(np.nanstd(y_error_list))
		print(np.nanstd(x_error_list))

		df = pd.DataFrame({'X Velocity Error (m/s)':x_error_list,'Y Velocity Error (m/s)':y_error_list})
		fig = px.density_heatmap(df, x='X Velocity Error (m/s)', y='Y Velocity Error (m/s)', marginal_x="histogram", marginal_y="histogram")
		fig.show()

	def plot_signal():
		x_list = []
		y_list = []
		for dummy in self.float_list:
			x_list+=dummy.ve.tolist()
			y_list+=dummy.vn.tolist()
		print(np.nanstd(y_list))
		print(np.nanstd(x_list))

		df = pd.DataFrame({'X Velocity (m/s)':x_list,'Y Velocity (m/s)':y_list})
		fig = px.density_heatmap(df, x='X Velocity (m/s)', y='Y Velocity (m/s)', marginal_x="histogram", marginal_y="histogram")
		fig.show()

	def return_observations(self):
		x_obs = []
		y_obs = []
		for dummy in self.float_list:
			x_obs+=dummy.ve.tolist()
			y_obs+=dummy.vn.tolist()
		print(np.nanstd(y_obs))
		print(np.nanstd(x_obs))
		return (x_obs,y_obs)

	def return_persistance(self):
		x_per = []
		y_per = []
		for dummy in self.float_list:
			x_per+=dummy.persistence_ve.tolist()
			y_per+=dummy.persistence_vn.tolist()
		print(np.nanstd(x_per))
		print(np.nanstd(y_per))
		return (x_per,y_per)

	def return_simulations(self):
		x_simulation = []
		y_simulation = []
		for dummy in self.float_list:
			x_simulation+=np.array(dummy.model_ve).tolist()
			y_simulation+=np.array(dummy.model_vn).tolist()
		print(np.nanstd(y_simulation))
		print(np.nanstd(x_simulation))
		return (x_simulation,y_simulation)



class HawaiiHYCOMAOMLFloats(AOMLFloatsBase):
	FloatClass = HawaiiHYCOMAOMLFloat
	path = os.path.join(get_data_folder(),'Raw/AOML/drifter_6hour_qc_3843_ec8c_723f.csv')

class HawaiiPACIOOSAOMLFloats(AOMLFloatsBase):
	FloatClass = HawaiiPACIOOSAOMLFloat
	path = os.path.join(get_data_folder(),'Raw/AOML/drifter_6hour_qc_3843_ec8c_723f.csv')

class HawaiiCopernicusAOMLFloats(AOMLFloatsBase):
	FloatClass = HawaiiCopernicusAOMLFloat
	path = os.path.join(get_data_folder(),'Raw/AOML/drifter_6hour_qc_3843_ec8c_723f.csv')


class PuertoRicoCopernicusAOMLFloats(AOMLFloatsBase):
	FloatClass = PuertoRicoCopernicusAOMLFloat
	path = os.path.join(get_data_folder(),'Raw/AOML/drifter_6hour_qc_4ee6_6c66_5c61.csv')

class PuertoRicoHYCOMAOMLFloats(AOMLFloatsBase):
	FloatClass = PuertoRicoHYCOMAOMLFloat
	path = os.path.join(get_data_folder(),'Raw/AOML/drifter_6hour_qc_4ee6_6c66_5c61.csv')

class MontereyCopernicusAOMLFloats(AOMLFloatsBase):
	FloatClass = MontereyCopernicusAOMLFloat
	path = os.path.join(get_data_folder(),'Raw/AOML/drifter_6hour_qc_2972_81ea_cf4e.csv')

class MontereyHYCOMAOMLFloats(AOMLFloatsBase):
	FloatClass = MontereyHYCOMAOMLFloat
	path = os.path.join(get_data_folder(),'Raw/AOML/drifter_6hour_qc_2972_81ea_cf4e.csv')


HYCOMPR = PuertoRicoHYCOMAOMLFloats()
HYCOMMonterey = MontereyHYCOMAOMLFloats()
HYCOMHawaii = HawaiiHYCOMAOMLFloats()
CopernicusPR = PuertoRicoCopernicusAOMLFloats()
CopernicusMonterey = MontereyCopernicusAOMLFloats()
CopernicusHawaii = HawaiiCopernicusAOMLFloats()
PACIOOSHawaii = HawaiiPACIOOSAOMLFloats()

def std_cor_calc(obs,sim):
	truth = [not (np.isnan(x) | np.isnan(y)) for x,y in zip(obs,sim)]
	cor = np.corrcoef(np.array(obs)[truth],np.array(sim)[truth])[0,1]
	return (cor, np.nanstd(obs),np.nanstd(sim))


x_pr_obs,y_pr_obs = HYCOMPR.return_observations()
x_pr_per,y_pr_per = HYCOMPR.return_persistance()
x_pr_hycom, y_pr_hycom = HYCOMPR.return_simulations()
x_pr_copernicus, y_pr_copernicus = CopernicusPR.return_simulations()


x_m_obs,y_m_obs = HYCOMMonterey.return_observations()
x_m_per,y_m_per = HYCOMMonterey.return_persistance()
x_m_hycom, y_m_hycom = HYCOMMonterey.return_simulations()
x_m_copernicus, y_m_copernicus = CopernicusMonterey.return_simulations()

x_h_obs,y_h_obs = HYCOMHawaii.return_observations()
x_h_per,y_h_per = HYCOMHawaii.return_persistance()
x_h_hycom, y_h_hycom = HYCOMHawaii.return_simulations()
x_h_copernicus, y_h_copernicus = CopernicusHawaii.return_simulations()
x_h_pacioos, y_h_pacioos = PACIOOSHawaii.return_simulations()


data_list = []
for obs,sim in [(x_pr_obs,x_pr_hycom),(y_pr_obs,y_pr_hycom),(x_m_obs,x_m_hycom),(y_m_obs,y_m_hycom),
(x_h_obs,x_h_hycom),(y_h_obs,y_h_hycom),(x_h_obs,x_h_pacioos),(y_h_obs,y_h_pacioos),(x_pr_obs,x_pr_per),
(y_pr_obs,y_pr_per),(x_m_obs,x_m_per),(y_m_obs,y_m_per),(x_h_obs,x_h_per),(y_h_obs,y_h_per),
(x_pr_obs,x_pr_copernicus),(y_pr_obs,y_pr_copernicus),(x_m_obs,x_m_copernicus),(y_m_obs,y_m_copernicus),
(x_h_obs,x_h_copernicus),(y_h_obs,y_h_copernicus),]:
	data_list.append(std_cor_calc(obs,sim))

observations = {
	'Puerto Rico X':{'std':data_list[0][1]},
	'Puerto Rico Y':{'std':data_list[1][1]},
	'Monterey X':{'std':data_list[2][1]},
	'Monterey Y':{'std':data_list[3][1]},
	'Hawaii X':{'std':data_list[4][1]},
	'Hawaii Y':{'std':data_list[5][1]}
}
predictions = {
	'Puerto Rico X':{'HYCOM':{'std':data_list[0][2],'corr_coeff':data_list[0][0]},
		'PACIOOS':{'std':100,'corr_coeff':data_list[6][0]},
		'Persistance':{'std':data_list[8][2],'corr_coeff':data_list[8][0]},
		'Copernicus':{'std':data_list[14][2],'corr_coeff':data_list[14][0]}},
	'Puerto Rico Y':{'HYCOM':{'std':data_list[1][2],'corr_coeff':data_list[1][0]},
		'PACIOOS':{'std':100,'corr_coeff':data_list[6][0]},
		'Persistance':{'std':data_list[9][2],'corr_coeff':data_list[9][0]},
		'Copernicus':{'std':data_list[15][2],'corr_coeff':data_list[15][0]}},
	'Monterey X':{'HYCOM':{'std':data_list[2][2],'corr_coeff':data_list[2][0]},
		'PACIOOS':{'std':100,'corr_coeff':data_list[6][0]},
		'Persistance':{'std':data_list[10][2],'corr_coeff':data_list[10][0]},
		'Copernicus':{'std':data_list[16][2],'corr_coeff':data_list[16][0]}},
	'Monterey Y':{'HYCOM':{'std':data_list[3][2],'corr_coeff':data_list[3][0]},
		'PACIOOS':{'std':100,'corr_coeff':data_list[6][0]},
		'Persistance':{'std':data_list[11][2],'corr_coeff':data_list[11][0]},
		'Copernicus':{'std':data_list[17][2],'corr_coeff':data_list[17][0]}},
	'Hawaii X':{'HYCOM':{'std':data_list[4][2],'corr_coeff':data_list[4][0]},
		'PACIOOS':{'std':data_list[6][2],'corr_coeff':data_list[6][0]},
		'Persistance':{'std':data_list[12][2],'corr_coeff':data_list[12][0]},
		'Copernicus':{'std':data_list[18][2],'corr_coeff':data_list[18][0]}},
	'Hawaii Y':{'HYCOM':{'std':data_list[5][2],'corr_coeff':data_list[5][0]},
		'PACIOOS':{'std':data_list[7][2],'corr_coeff':data_list[7][0]},
		'Persistance':{'std':data_list[13][2],'corr_coeff':data_list[13][0]},
		'Copernicus':{'std':data_list[19][2],'corr_coeff':data_list[19][0]}}
}

rects = {'Puerto Rico X':321, 'Puerto Rico Y':322, 'Monterey X':323, 'Monterey Y':324, 'Hawaii X':325, 'Hawaii Y':326}
_ = taylor_plot(observations=observations,
            simulations=predictions,figsize=(20,20),
            axis_locs=rects,show=False)
_.savefig('Taylor')