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
            particle.cycle_phase = 2
    elif particle.cycle_phase == 2:
        # Phase 2: Sinking further to max_depth
        particle.depth += particle.vertical_speed * particle.dt
        if particle.depth >= particle.max_depth:
            particle.depth = particle.max_depth
            particle.cycle_phase = 3
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