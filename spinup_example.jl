using SpeedyWeather

# constants
n_particles = 10
spinup_days = 30
days_after_spinup = 300

# Setup grid, particle advection scheme, model, and simulation
spectral_grid = SpectralGrid(nparticles=n_particles)
particle_advection = ParticleAdvection2D(spectral_grid, layer=8)
model = PrimitiveWetModel(spectral_grid; particle_advection)
simulation = initialize!(model)

# Spin up the model for spinup_days
run!(simulation, period=Day(spinup_days))

# Add particle tracker
particle_tracker = ParticleTracker(spectral_grid)
add!(model, :particle_tracker => particle_tracker)

# Adjust initial locations of particles
(; particles) = simulation.prognostic_variables
particles .= rand(Particle, n_particles)

# then run! simulation after spinup
run!(simulation, period=Day(days_after_spinup))
