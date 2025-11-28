using SpeedyWeather, TravellingSailorProblem, GLMakie, LinearAlgebra, NCDatasets

### TODO THIS VERSION DOES NOT WORK


### Constants and parameters ###

ntrunc = 40 # Spectral grid truncation level
nlayers = 8 # Number of layers in the atmosphere
particle_layer = 8 # Layer in which the particles are released
dist_km = 10 # Perturbation in initial position in km
grid_type = HEALPixGrid # Type of grid to use

# Spinup parameters
spinup_days = 50 # Spinup length TODO How long to use???
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



### Calculate FTLE ###

# Read in particle positions
particles_ds = Dataset("particles.nc","r")

# Number of times for which particle position was saved
Ntime = particles_ds.dim["time"]

# Initialise array to hold FTLE
FTLE_grid_time = Array{Float64}(undef, spectral_grid.npoints, Ntime) 

# Iterate over time steps
for tindex in 1:Ntime

    plonds = particles_ds["lon"][:,tindex]
    platds = particles_ds["lat"][:,tindex]

    # !! TODO need to figure out time units

    # Calculate displacement gradient matrix
    B = displacement_gradient_matrix(plonds, platds, dist_km)

    FTLE_grid_time[:,tindex] .= FTLE_over_grid(B, tindex)

end

close(particles_ds)


### Visualise ###

grid = spectral_grid.grid

function tslice_heatmap(tindex)
    FTLE_grid = Field(FTLE[:,tindex], grid);
    heatmap(FTLE_grid)
end
