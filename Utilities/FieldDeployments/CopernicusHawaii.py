from GeneralUtilities.Plot.Cartopy.regional_plot import KonaCartopy
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.FieldDeployments.FieldDeploymentBase import mean_monthly_plot,quiver_movie,shear_movie,eke_plots,pdf_particles_compute
from HyperNav.Utilities.Compute.RunParcels import UVPrediction,ParticleDataset,ParticleList,ClearSky
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
import cartopy.crs as ccrs
import numpy as np
import os
from HyperNav.Utilities.Compute.ArgoBehavior import ArgoVerticalMovement
from GeneralUtilities.Plot.Cartopy.regional_plot import RegionalBase
from HyperNav.Utilities.Data.CopernicusGlobal import HawaiiCopernicus
from HyperNav.Utilities.Utilities import HyperNavBehavior

file_handler = FilePathHandler(ROOT_DIR,'CopernicusHawaiiFutureDeployment')


class FutureHawaiiCartopy(RegionalBase):
    llcrnrlon=-158
    llcrnrlat=18
    urcrnrlon=-156
    urcrnrlat=20
    def __init__(self,*args,**kwargs):
        print('I am plotting Kona')
        super().__init__(*args,**kwargs)

def future_prediction():
	date_start = datetime.datetime(2022,5,7)
	date_end = datetime.datetime(2022,5,12)
	uv_class = HawaiiCopernicus.load(date_start-datetime.timedelta(days=2),date_end+datetime.timedelta(days=2))
	data,dimensions = uv_class.return_parcels_uv(date_start-datetime.timedelta(days=2),date_end+datetime.timedelta(days=2))
	start_lat = 19.0536
	start_lon = -156.7926
	out_list = []
	for drift_depth in [500,250,10]:
		behavior_class = HyperNavBehavior(drift_depth,start_lat,start_lon,date_start,date_end)
		for label,behavior in [behavior_class.no_hypernav(),behavior_class.normal(),behavior_class.long_trans(),behavior_class.two_day_cycle()]:
			pl = ParticleList()
			prediction = UVPrediction(behavior,data,dimensions)
			prediction.create_prediction()
			nc = ParticleDataset('/Users/paulchamberlain/Projects/HyperNav/Pipeline/Compute/RunParcels/tmp/Uniform_out.nc')
			pl.append(nc)
			out_list.append((label,drift_depth,pl))
	plt.rcParams["figure.figsize"] = (15,15)
	DUM,DUM,ax = FutureHawaiiCartopy().get_map()
	for mission,depth,pl in out_list:
		lats = pl[0]['lat'][:].mean(axis=0)
		lons = pl[0]['lon'][:].mean(axis=0)
		ax.plot(lons,lats,zorder=15,label=(str(depth)+' '+mission))
	plt.legend()
	plt.savefig('out')


	plt.savefig(file_handler.out_file('deployment_movie/'+str(r)))
	plt.close()
	os.chdir(file_handler.out_file('deployment_movie'))
	os.system("ffmpeg -r 5 -i %01d.png -vcodec mpeg4 -y movie.mp4")