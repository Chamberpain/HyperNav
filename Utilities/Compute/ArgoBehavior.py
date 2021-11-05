

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
	cycletime = 2 * (86400-driftdepth/vertical_speed)  # total time of cycle in seconds
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
	cycletime = 1 * (21*3600-driftdepth/vertical_speed)  # total time of cycle in seconds
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