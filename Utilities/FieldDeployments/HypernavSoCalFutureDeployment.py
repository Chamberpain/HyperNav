from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots,pdf_particles_compute
from HyperNav.Utilities.Compute.RunParcels import create_prediction,ParticleDataset,ParticleList
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
import cartopy.crs as ccrs
import numpy as np
import os
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
from HyperNav.Utilities.Data.HYCOM import HYCOMSouthernCalifornia
file_handler = FilePathHandler(ROOT_DIR,'HypernavSoCalFutureDeployment')


class SoCalCartopy(RegionalBase):
    llcrnrlon=-120
    llcrnrlat=32
    urcrnrlon=-116.7
    urcrnrlat=33.5
    def __init__(self,*args,**kwargs):
        print('I am plotting Southern California')
        super().__init__(*args,**kwargs)

class HYCOMFutureSoCal(HYCOMSouthernCalifornia):
	PlotClass = SoCalCartopy
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)


def socal_mean_monthly_plot():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2014,1,1),datetime.datetime(2022,1,1))
	mean_monthly_plot(uv_class,file_handler,month=12)


def socal_quiver_movie():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2014,1,1),datetime.datetime(2022,1,1))
	mask = [(x>datetime.datetime(2020,12,1))&(x<datetime.datetime(2020,12,30)) for x in uv_class.time]
	quiver_movie(uv_class,mask,file_handler)


def socal_shear_movie():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2014,1,1),datetime.datetime(2022,1,1))
	lat = 32.9
	lon = -117.8
	mask = [(x>datetime.datetime(2020,12,1))&(x<datetime.datetime(2020,12,30)) for x in uv_class.time]
	shear_movie(uv_class,mask,file_handler,lat,lon)

def socal_eke():
	uv_class = HYCOMFutureSoCal.load(datetime.datetime(2014,1,1),datetime.datetime(2022,1,1))
	eke_plots(uv_class,file_handler)

def monterey_particles_compute():
	uv_class = HYCOMFutureSoCal
	float_list = [({'lat':32.9,'lon':-117.8,'time':datetime.datetime(2015,11,20)},'site_1')]
	pdf_particles_compute(uv_class,float_list,file_handler)
