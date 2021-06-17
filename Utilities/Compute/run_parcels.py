from parcels import DiffusionUniformKh, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4,AdvectionDiffusionM1, plotTrajectoriesFile
import numpy as np
from datetime import timedelta
from operator import attrgetter
from HyperNav.Utilities.Data.data_parse import raw_base_file, processed_base_file
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from geopy import distance
from GeneralUtilities.Data.depth.depth_utilities import PACIOOS as Depth
from parcels import ParcelsRandom as random
import os




def add_list(list_):
    holder = []
    for item in list_:
        holder += [item + dummy for dummy in np.random.normal(scale=.1,size=particle_num)]
    return holder

def PastArgoMovement0055(particle,fieldset,time):
    
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10
    vertical_speed = 0.10  # sink and rise speed in m/s

    for k in range(len(prediction_list)-1):
        driftdepth = depth_list[k]  # maximum depth in m
        cycletime = 1 * (time_delta[k]-driftdepth/vertical_speed)  # total time of cycle in seconds



    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age







def PastArgoMovement0054(particle,fieldset,time):
    prediction_list = SiteAPI.get_past_locations('0054')

    for k in range(len(prediction_list)-1):






# Define the new Kernel that mimics Argo vertical movement
def ArgoVerticalMovement700(particle, fieldset, time):
    driftdepth = 700  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age


def ArgoVerticalMovement600(particle, fieldset, time):
    driftdepth = 600  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement500(particle, fieldset, time):
    driftdepth = 500  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement400(particle, fieldset, time):
    driftdepth = 400  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age


def ArgoVerticalMovement300(particle, fieldset, time):
    driftdepth = 300  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement200(particle, fieldset, time):
    driftdepth = 200  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement100(particle, fieldset, time):
    driftdepth = 100  # maximum depth in m
    vertical_speed = 0.1  # sink and rise speed in m/s
    cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age

def ArgoVerticalMovement50(particle, fieldset, time):
    driftdepth = 50  # maximum depth in m
    vertical_speed = 0.1  # sink and rise speed in m/s
    cycletime = 1 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
    surftime = 2 * 3600  # time of deep drift in seconds
    mindepth = 10

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.cycle_age += particle.dt
        if particle.cycle_age >= cycletime:
            particle.cycle_age = 0  # reset cycle_age for next cycle
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        particle.cycle_age += particle.dt
        #particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth <= mindepth:
            particle.depth = mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.surf_age = 0
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        particle.cycle_age += particle.dt
        particle.surf_age += particle.dt
        if particle.surf_age > surftime:
            particle.cycle_phase = 0

    if particle.state == ErrorCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age


class ArgoParticle(JITParticle):
    # Phase of cycle: init_descend=0, drift=1, profile_descend=2, profile_ascend=3, transmit=4
    cycle_phase = Variable('cycle_phase', dtype=np.int32, initial=0.)
    cycle_age = Variable('cycle_age', dtype=np.float32, initial=0.)
    drift_age = Variable('surf_age', dtype=np.float32, initial=0.)
    profile_idx = Variable('profile_idx', dtype=np.float32, initial=0.)
    #temp = Variable('temp', dtype=np.float32, initial=np.nan)  # if fieldset has temperature

prediction_list_54 = SiteAPI.get_past_locations('0054')
depth_list_54 = prediction_list_54.return_depth()[1:]
time_delta_54 = [(time_list[k+1]-time_list[k]).seconds for k in range(len(prediction_list_54.return_time())-1)]
class HyperNav54(ArgoParticle)
    k = Variable('k', dtype=np.float32, initial=0.)

prediction_list_55 = SiteAPI.get_past_locations('0055')
depth_list_55 = prediction_list_55.return_depth()[1:]
time_delta_55 = [(time_list[k+1]-time_list[k]).seconds for k in range(len(prediction_list_55.return_time())-1)]
class HyperNav55(ArgoParticle)
    k = Variable('k', dtype=np.float32, initial=0.)

def get_test_particles():
    return ParticleSet.from_list(fieldset,
                                 pclass=ArgoParticle,
                                 lat=np.array(add_list(lat_list)),
                                 lon=np.array(add_list(lon_list)),
                                 time=np.zeros(particle_num*len(lat_list)),
                                 depth=[10]*particle_num*len(lat_list)
                                 )
def run_parcels():
    particle_num = 300
    lat_list = [20.7,20.5,20.9,19.7,19.4,20.0]
    lon_list = [-157.3,-157.3,-157.4,-156.2,-156.1,-156.1]
    depth_700 = 18

    K_bar = 0.000000000025

    dimensions = {'lat': 'lat',
    'lon':'lon',
    'time':'time',
    'depth':'depth'}

    variables = {'U':'u','V':'v'}

    files = os.listdir(raw_base_file)



    for file in files:
        if not file.endswith('.nc'):
            continue
        filenames = {'U':raw_base_file+file,
        'V':raw_base_file+file}

        fieldset = FieldSet.from_netcdf(filenames,variables,dimensions)
        fieldset.add_constant('Kh_meridional',K_bar)
        fieldset.add_constant('Kh_zonal',K_bar)
        testParticles = get_test_particles()
        kernels = testParticles.Kernel(ArgoVerticalMovement)
        dt = 15 #5 minute timestep
        output_file = testParticles.ParticleFile(name=processed_base_file+file.split('.')[0]+"_Uniform_out.nc",
                                                 outputdt=timedelta(minutes=dt))
        testParticles.execute(kernels,
                              runtime=timedelta(days=28),
                              dt=timedelta(minutes=dt),
                              output_file=output_file,)
        output_file.export()
        output_file.close()


    start_lat_list=[]
    start_lon_list=[]
    end_lat_list=[]
    end_lon_list=[]
    total_depth_flag_list = []
    depth = Depth()
    files = os.listdir(processed_base_file)
    kona_pos = (19.6400,-155.9969)
    lahaina_pos = (20.8700,-156.68)

    for file in files:
        if not file.endswith('.nc'):
            continue
        nc = Dataset(processed_base_file+file)

        float_depth_flag_list = []
        for k in range(nc.variables["lon"][:].shape[0]):
            print(k)
            lats = nc.variables["lat"][:][k,:].data
            lons = nc.variables["lon"][:][k,:].data

            start_lat_list.append(lats[0])
            start_lon_list.append(lons[0])
            end_lat_list.append(lats[-1])
            end_lon_list.append(lons[-1])

            for x in list(zip(lats[::10],lons[::10])):
                print(depth.return_z(x))
                depth_truth = depth.too_shallow(x)
                if depth_truth:
                    break
            if depth_truth:
                float_depth_flag_list.append(False)
                print(float_depth_flag_list[-1])
                continue
            float_depth_flag_list.append(True)
            print(float_depth_flag_list[-1])
        total_depth_flag_list += float_depth_flag_list
        nc.close()
        # plotTrajectoriesFile(processed_base_file+"Uniform_out.nc")
        # testParticles.show(field=fieldset.V,depth_level=depth_700,time=0)

    # x = nc.variables["lon"][:][0,:].squeeze()
    # y = nc.variables["lat"][:][0,:].squeeze()
    # z = nc.variables["z"][:][0,:].squeeze()
    # fig = plt.figure(figsize=(13,10))
    # ax = plt.axes(projection='3d')
    # cb = ax.scatter(x, y, z, c=z, s=20, marker="o")
    # ax.set_xlabel("Longitude")
    # ax.set_ylabel("Latitude")
    # ax.set_zlabel("Depth (m)")
    # ax.set_zlim(np.max(z),0)
    # plt.savefig(processed_base_file+'z_prof_example')
    # plt.close()
    files = os.listdir(raw_base_file)
    for file in files:
        if not file.endswith('.nc'):
            continue
        model_nc = Dataset(raw_base_file+file)
        lon_grid = model_nc['lon'][:]
        lat_grid = model_nc['lat'][:]
    XX,YY,m = basemap_setup(lat_grid,lon_grid,'Moby')

    nx = 50
    ny = 50

    lon_bins = np.linspace(lon_grid.min(), lon_grid.max(), nx+1)
    lat_bins = np.linspace(lat_grid.min(), lat_grid.max(), ny+1)

    density, _, _ = np.histogram2d(start_lat_list,start_lon_list, [lat_bins, lon_bins])
    density = np.ma.masked_equal(density,0)
    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    xs, ys = m(lon_bins_2d, lat_bins_2d)
    plt.pcolormesh(xs, ys, density)
    plt.colorbar(orientation='horizontal',label='Number of Floats Deployed in Bin')
    plt.scatter(*m(end_lon_list,end_lat_list),s=0.4,c='k',alpha=0.4)
    plt.savefig(processed_base_file+'density_plot')
    plt.close()


    dist_list_kona = [distance.great_circle(kona_pos,x).nm for x in zip(end_lat_list,end_lon_list)]
    dist_list_lahaina = [distance.great_circle(lahaina_pos,x).nm for x in zip(end_lat_list,end_lon_list)]

    kona_dist_flag_list = [(kona<50) for kona in dist_list_kona]
    lahaina_dist_flag_list = [(lahaina<50) for lahaina in dist_list_lahaina]
    total_dist_flag_list = [(kona<50)|(lahaina<50) for kona,lahaina in zip(dist_list_kona,dist_list_lahaina)]

    lats_bin = depth.y.data
    lons_bin = depth.x.data

    result = scipy.stats.binned_statistic_2d(start_lon_list,start_lat_list,total_depth_flag_list,statistic='mean',bins=[lons_bin[400:800],lats_bin[200:600]])
    YY,XX = np.meshgrid(result.y_edge[:-1],result.x_edge[:-1])
    XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
    m.pcolormesh(XX,YY,(1-result.statistic.T)*100)
    plt.colorbar(label='Chance of becoming bathymetric sensor (%)')
    plt.title('Map of mean grounding chance')
    plt.savefig(processed_base_file+'moby_grounding_map')
    plt.close()
    result = scipy.stats.binned_statistic_2d(start_lon_list,start_lat_list,dist_list_kona,statistic='mean',bins=[lons_bin,lats_bin])
    YY,XX = np.meshgrid(result.y_edge[:-1],result.x_edge[:-1])
    XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
    m.pcolormesh(XX,YY,(result.statistic.T))
    plt.colorbar(label='Distance from Kona at end of run (nm)')
    plt.title('Map of mean distance')
    plt.savefig(processed_base_file+'moby_kona_distance_map')
    plt.close()
    mask = np.array(total_depth_flag_list)&np.array(kona_dist_flag_list)
    lons = np.array(start_lon_list)[mask]
    lats = np.array(start_lat_list)[mask]
    XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
    m.scatter(lons,lats,latlon=True)
    plt.title('Deployment locations of successful particles for Kona')
    plt.savefig(processed_base_file+'moby_kona_success map')
    plt.close()
    result = scipy.stats.binned_statistic_2d(start_lon_list,start_lat_list,dist_list_lahaina,statistic='mean',bins=[lons_bin,lats_bin])
    YY,XX = np.meshgrid(result.y_edge[:-1],result.x_edge[:-1])
    XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
    m.pcolormesh(XX,YY,(result.statistic.T))
    plt.colorbar(label='Distance from Lahaina at end of run (nm)')
    plt.title('Map of mean distance')
    plt.savefig(processed_base_file+'moby_lahaina_distance_map')
    plt.close()
    mask = np.array(total_depth_flag_list)&np.array(lahaina_dist_flag_list)
    lons = np.array(start_lon_list)[mask]
    lats = np.array(start_lat_list)[mask]
    XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
    m.scatter(lons,lats,latlon=True)
    plt.title('Deployment locations of successful particles for Lahaina ')
    plt.savefig(processed_base_file+'moby_lahaina_success map')
    plt.close()
    mask = np.array(total_depth_flag_list)&np.array(total_dist_flag_list)
    lons = np.array(start_lon_list)[mask]
    lats = np.array(start_lat_list)[mask]
    XX,YY,m = basemap_setup(result.y_edge[:-1],result.x_edge[:-1],'Moby')
    m.scatter(lons,lats,latlon=True)
    plt.title('All successful deployement locations')
    plt.savefig(processed_base_file+'moby_total_success map')
    plt.close()
    result = scipy.stats.binned_statistic_2d(start_lon_list,start_lat_list,mask,statistic='mean',bins=[lons_bin[400:800][::4],lats_bin[200:600][::4]])
    YY,XX = np.meshgrid(result.y_edge[:-1][::4],result.x_edge[:-1][::4])
    XX,YY,m = basemap_setup(result.y_edge[:-1][::4],result.x_edge[:-1][::4],'Moby')
    m.pcolormesh(XX,YY,(1-result.statistic.T)*100)
    plt.colorbar(label='Chance of Success (%)')
    plt.title('Map of successful deployments')
    plt.savefig(processed_base_file+'moby_success_map')
    plt.close()

