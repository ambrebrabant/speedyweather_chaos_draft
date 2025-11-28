using SpeedyWeather, TravellingSailorProblem, GLMakie, LinearAlgebra, NCDatasets

include("FTLE_functions.jl")



### Constants and parameters ###

ntrunc = 40 # Spectral grid truncation level
nlayers = 8 # Number of layers in the atmosphere
particle_layer = 8 # Layer in which the particles are released
dist_km = 10 # Perturbation in initial position in km
grid_type = HEALPixGrid # Type of grid to use

# Spinup parameters
spinup_days = 50 # Spinup length TODO How long to use?
days_after_spinup = 10 # Simulation length after spinup



### Setup ###

# Setup grid
spectral_grid = setup_grid_FTLE(ntrunc, nlayers, grid_type)

# Set up particle advection scheme, model, and simulation
particle_advection = ParticleAdvection2D(spectral_grid, layer=particle_layer)
model = PrimitiveWetModel(spectral_grid; particle_advection)
simulation = initialize!(model)

# Spin up the model for spinup_days
run!(simulation, period=Day(spinup_days))

# Add particle tracker
particle_tracker = ParticleTracker(spectral_grid)
add!(model, :particle_tracker => particle_tracker)



### Perturb initial locations of particles ###

londs, latds = RingGrids.get_londlatds(spectral_grid.grid)
(; particles) = simulation.prognostic_variables
perturb_positions_FTLE(particles, londs, latds, dist_km)



### Run ###
run!(simulation, period=Day(days_after_spinup))



### Calculate FTLE from positions at last time step ###

plonds = [part.lon for part in particles]
platds = [part.lat for part in particles]

B = displacement_gradient_matrix(plonds, platds, dist_km)

FTLE_grid = FTLE_over_grid(B, days_after_spinup)


### Visualise ###

grid = spectral_grid.grid
heatmap(Field(FTLE_grid, grid))

