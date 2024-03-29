import pickle
from datetime import datetime, timedelta

import numpy as np
from parcels import JITParticle, Variable, FieldSet, ParticleSet, AdvectionRK4
from netCDF4 import Dataset


def find_nearest(array, value):
    """
    Find nearest neighbor and return its index
    :param array:
    :param value:
    :return:
    """
    idx = (np.abs(array - value)).argmin()
    return int(idx)


class ArgoParticle(JITParticle):
    # Phase of cycle: init_descend=0, drift=1, profile_descend=2, profile_ascend=3, transmit=4
    cycle_phase = Variable('cycle_phase', dtype=np.int32, initial=0., to_write=False)
    cycle_age = Variable('cycle_age', dtype=np.float32, initial=0., to_write=False)
    surface_age = Variable('surface_age', dtype=np.float32, initial=0., to_write=False)
    drift_depth = Variable('drift_depth', dtype=np.int32,  to_write=False)  # drifting depth in m
    min_depth = Variable('min_depth', dtype=np.int32, to_write=False)       # shallowest depth in m
    # max_depth = Variable('max_depth', dtype=np.int32, to_write=False)      # profile depth in m
    vertical_speed = Variable('vertical_speed', dtype=np.float32, to_write=False)  # sink and rise speed in m/s  (average speed of profile 0054.21171)
    surface_time = Variable('surface_time', dtype=np.int32, to_write=False)        # surface time in seconds
    cycle_time = Variable('cycle_time', dtype=np.float32, to_write=False)          # total time of cycle in seconds


def ArgoVerticalMovement(particle, fieldset, time):
    particle.cycle_age += particle.dt
    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is drift_depth
        particle.depth += particle.vertical_speed * particle.dt
        if particle.depth >= particle.drift_depth:
            particle.depth = particle.drift_depth
            particle.cycle_phase = 1
    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drift_time seconds
        if particle.cycle_age >= particle.cycle_time:
            # if particle.max_depth > particle.drift_depth:
            #     particle.cycle_phase = 2
            # else:
            particle.cycle_phase = 3
    # elif particle.cycle_phase == 2:
    #     # Phase 2: Sinking further to max_depth
    #     particle.depth += particle.vertical_speed * particle.dt
    #     if particle.depth >= particle.max_depth:
    #         particle.depth = particle.max_depth
    #         particle.cycle_phase = 3
    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= particle.vertical_speed * particle.dt
        if particle.depth <= particle.min_depth:
            particle.depth = particle.min_depth
            particle.surface_age = 0
            particle.cycle_phase = 4
    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycle_time is reached
        particle.surface_age += particle.dt
        if particle.surface_age >= particle.surface_time:
            particle.cycle_age = 0
            particle.cycle_phase = 0


def run_prediction(field, particle, dt_of_interest, wk_filename='particles.nc', n_particles=500,
                   run_time=timedelta(days=3), output_time_step=timedelta(minutes=15)):
    # Set Parcels UVW Field (assume depth is positive down)
    field_set = FieldSet.from_data(field['data'], field['dimensions'], transpose=False)
    field_set.mindepth = field['dimensions']['depth'][0]
    field_set.add_constant('Kh_meridional', 0.000000000025)
    field_set.add_constant('Kh_zonal', 0.000000000025)
    # Set Particles
    particles = dict(lat=particle['lat'] + np.random.normal(scale=.05, size=n_particles),
                     lon=particle['lon'] + np.random.normal(scale=.05, size=n_particles),
                     time=np.array([particle['time']] * n_particles),
                     depth=np.array([particle['depth']] * n_particles),
                     min_depth=np.array([particle['min_depth'] if 'min_depth' in particle.keys() else 10] * n_particles, dtype=np.int32),
                     drift_depth=np.array([particle['drift_depth'] if 'drift_depth' in particle.keys() else 500] * n_particles, dtype=np.int32),
                     vertical_speed=np.array([particle['vertical_speed'] if 'vertical_speed' in particle.keys() else 0.076] * n_particles, dtype=np.float32),
                     surface_time=np.array([particle['surface_time'] if 'surface_time' in particle.keys() else 2 * 3600] * n_particles, dtype=np.int32),
                     cycle_time=np.array([particle['total_cycle_time'] if 'total_cycle_time' in particle.keys() else 2 * 86400] * n_particles, dtype=np.float32))
    particles['cycle_time'] -= (particles['drift_depth'] / particles['vertical_speed']) + particles['surface_time']
    particle_set = ParticleSet.from_list(field_set, pclass=ArgoParticle, **particles)
    kernels = ArgoVerticalMovement + particle_set.Kernel(AdvectionRK4)
    # Run Particle Simulation
    particle_set.execute(kernels, runtime=run_time, dt=timedelta(minutes=5),  # step to compute
                         output_file=particle_set.ParticleFile(name=wk_filename, outputdt=output_time_step))
    # Average Final Positions
    with Dataset(wk_filename) as data:
        ts = data.variables['time'][:][0, :]
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
    results = {k: [] for k in ['dt', 'lat', 'lat_se', 'lon', 'lon_se']}
    for dt in dt_of_interest:
        idx = find_nearest(ts, dt.timestamp())
        results['dt'].append(datetime.fromtimestamp(ts[idx]))
        results['lat'].append(lat[:, idx].mean())
        results['lat_se'].append(lat[:, idx].std() / np.sqrt(n_particles))
        results['lon'].append(lon[:, idx].mean())
        results['lon_se'].append(lon[:, idx].std() / np.sqrt(n_particles))
    return results


if __name__ == '__main__':
    import os

    root = '/Users/nils/Data/HyperNAV/TrajectoryPrediction/Hawaii'
    # UV Field
    path_to_field = os.path.join(root, '2021102212Z.pickle')
    with open(path_to_field, 'rb') as f:
        field = pickle.load(f)
    # Argo Float Position and Profiling Parameters
    dt = datetime(2021, 10, 22, 12, 0, 0)
    # dt = datetime.utcnow()
    argo_cfg = {'lat': 19.3127, 'lon': -156.1781 % 360, 'time': dt.timestamp(), 'depth': 10,
                'min_depth': 10, 'drift_depth': 500, 'surface_time': 2 * 3600, 'total_cycle_time': 1 * 86400}
    # Date & times of interest
    dt_of_interest = [dt + timedelta(days=d) for d in range(1, 4)]
    # Run prediction
    path_to_prediction = os.path.join(root, 'prediction.nc')
    positions = run_prediction(field, argo_cfg, dt_of_interest, path_to_prediction, n_particles=1)

    # Plot results
    import netCDF4
    import plotly.graph_objects as go

    nc = netCDF4.Dataset(path_to_prediction)
    ts = nc.variables["time"][:].squeeze()
    x = nc.variables["lon"][:].squeeze()
    y = nc.variables["lat"][:].squeeze()
    z = nc.variables["z"][:].squeeze()
    nc.close()

    fig = go.Figure()
    if len(x.shape) == 1:
        fig.add_scatter3d(x=x, y=y, z=z, marker_color=ts, text=[datetime.fromtimestamp(t) for t in ts])
    else:
        for k in range(len(x)):
            fig.add_scatter3d(x=x[k], y=y[k], z=z[k], marker_color=ts[k], text=[datetime.fromtimestamp(t) for t in ts[k]])
    fig.update_layout(scene=dict(xaxis_title='lon', yaxis_title='lat', zaxis_title='depth',
                                 zaxis_autorange='reversed'))
    fig.show()
