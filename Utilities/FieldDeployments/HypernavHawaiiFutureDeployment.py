from HyperNav.Utilities.Data.HYCOM import HYCOMHawaii
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots
from HyperNav.Utilities.Compute.RunParcels import UVPrediction,ParticleDataset,ParticleList
import cartopy.crs as ccrs
import numpy as np
import os
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement700,ArgoVerticalMovement600,ArgoVerticalMovement500,ArgoVerticalMovement400,ArgoVerticalMovement300,ArgoVerticalMovement200,ArgoVerticalMovement100,ArgoVerticalMovement50
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
file_handler = FilePathHandler(ROOT_DIR,'HypernavHawaiiFutureDeployment')


class FutureHawaiiCartopy(RegionalBase):
    llcrnrlon=-157.5
    llcrnrlat=18.5
    urcrnrlon=-154.5
    urcrnrlat=21.5
    def __init__(self,*args,**kwargs):
        print('I am plotting Kona')
        super().__init__(*args,**kwargs)

def add_zero_boundary_conditions(_matrix):
		_matrix[:,:,:,0] = 0
		_matrix[:,:,:,-1] = 0
		_matrix[:,:,0,:] = 0
		_matrix[:,:,-1,:] = 0
		return _matrix

class HYCOMFutureHawaii(HYCOMHawaii):
	PlotClass = FutureHawaiiCartopy
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.u = add_zero_boundary_conditions(self.u)
		self.v = add_zero_boundary_conditions(self.v)		

def hawaii_mean_monthly_plot():
	uv_class = HYCOMFutureHawaii.load()
	mean_monthly_plot(uv_class,file_handler,month=12)


def hawaii_quiver_movie():
	uv_class = HYCOMFutureHawaii.load()
	mask = [(x>datetime.datetime(2020,12,1))&(x<datetime.datetime(2020,12,30)) for x in uv_class.time]
	quiver_movie(uv_class,mask,file_handler)


def hawaii_shear_movie():
	uv_class =  HYCOMFutureHawaii.load()
	lat = 20
	lon = -156.1
	mask = [(x>datetime.datetime(2020,12,1))&(x<datetime.datetime(2020,12,30)) for x in uv_class.time]
	shear_movie(uv_class,mask,file_handler,lat,lon)

def hawaii_eke():
	uv_class =  HYCOMFutureHawaii.load()
	eke_plots(uv_class,file_handler)

def hawaii_particles_compute():
	uv_class = HYCOMFutureHawaii.load()
	float_list = [({'lat':19.5,'lon':-156.3,'time':datetime.datetime(2015,11,20)},'site_1')]
	pdf_particles_compute(uv_class,float_list,file_handler)
