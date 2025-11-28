using SpeedyWeather, GLMakie, CairoMakie, GeoMakie, LinearAlgebra, NCDatasets

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

### Compute the right Cauchy-Green deformation tensors
plonds = [part.lon for part in particles]
platds = [part.lat for part in particles]

B = displacement_gradient_matrix_central(plonds, platds, dist_km)

CGs = Vector{Matrix{Float64}}(undef, size(B,3))
min_λ = Vector{Float64}(undef, size(B,3))
max_λ = Vector{Float64}(undef, size(B,3))
min_ξ_u = Vector{Float64}(undef, size(B,3))
min_ξ_v = Vector{Float64}(undef, size(B,3))
max_ξ_u = Vector{Float64}(undef, size(B,3))
max_ξ_v = Vector{Float64}(undef, size(B,3))

for k in 1:size(B,3)
    CG = B[:,:,k]' * B[:,:,k]
    CGs[k] = CG
    F = eigen(CG)
    min_idx = argmin(F.values)
    max_idx = argmax(F.values)
    min_λ[k] = F.values[min_idx]
    max_λ[k] = F.values[max_idx]
    min_ξ_u[k] = F.vectors[1, min_idx]
    min_ξ_v[k] = F.vectors[2, min_idx]
    max_ξ_u[k] = F.vectors[1, max_idx]
    max_ξ_v[k] = F.vectors[2, max_idx]
end

# Turn the ξ into a Field, then a callable function
field_max_ξ_u = Field(max_ξ_u, spectral_grid.grid)
field_max_ξ_v = Field(max_ξ_v, spectral_grid.grid)
function max_ξ_itp(lon, lat)
    u = SpeedyWeather.interpolate(lon, lat, field_max_ξ_u)
    v = SpeedyWeather.interpolate(lon, lat, field_max_ξ_v)
    return Point2f(u, v)
end
field_min_ξ_u = Field(min_ξ_u, spectral_grid.grid)
field_min_ξ_v = Field(min_ξ_v, spectral_grid.grid)
function min_ξ_itp(lon, lat)
    u = SpeedyWeather.interpolate(lon, lat, field_min_ξ_u)
    v = SpeedyWeather.interpolate(lon, lat, field_min_ξ_v)
    return Point2f(u, v)
end


fig, ax, plt = streamplot(
    min_ξ_itp, 0..360, -90..90;
    color = t -> :red,
    gridsize = (16, 16),
    maxsteps = 2000,
    arrow_size = 0,
    axis = (;
        type = GeoAxis,
        title = "Streamplot of ξ₁ (red) and ξ₂ (blue)",
    )
)
streamplot!(ax, max_ξ_itp, 0..360, -90..90;
    color = t -> :blue,
    gridsize = (16, 16),
    maxsteps = 2000,
    arrow_size = 0,
)
fig
