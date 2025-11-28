using SpeedyWeather, TravellingSailorProblem, GLMakie, LinearAlgebra, NCDatasets

Re = 6.371e6 # Average Earth radius in meters

function FTLE_from_eigenvalue(lmax, T)
    """
    Calculates the FTLE from a time interval and a maximum eigenvalue

    Inputs:
        lmax: maximum eigenvalue of the right Cauchy-Green deformation tensor
        T: time interval over which lmax was calculated
    """
    return log(lmax)/2/T 
end

function setup_grid_FTLE(ntrunc, nlayers, grid_type)
    """
    Sets up a SpectralGrid with 4 particles per grid cell for calculating the FTLE

    ntrunc: truncation degree of the spectral grid
    nlayers: number of layers in the model atmosphere
    grid_type: grid type to use e.g. HEALPixGrid
    """
    spectral_grid = SpectralGrid(trunc=ntrunc, nlayers=nlayers, Grid=grid_type)
    nparticles = 4*spectral_grid.npoints # Number of particles
    spectral_grid = SpectralGrid(trunc=ntrunc, nlayers=nlayers, nparticles=nparticles, Grid=grid_type)
    return spectral_grid
end

function perturb_positions_FTLE(particles, londs, latds, dist_km)
    """
    Sets up the initial positions of particles for calculating the FTLE
    
    !! Modifies the simulation object in place

    particles: vector containing all Particle objects in the simulation
    londs: longitudes of grid cells
    latds: latitudes of grid cells
    dist_km: perturbation to apply in km
    """

    Npoints = length(londs) # Number of grid points

    cos_factor = cos.(deg2rad.(latds)) # Cos(latitude)

    del_lat = rad2deg((dist_km * 1000 / Re)) # Latitude perturbation in degrees

    # Perturbed East/West
    particles[1:4:end] .= [Particle(londs[i] + del_lat/cos_factor[i], latds[i]) for i in 1:Npoints]
    particles[2:4:end] .= [Particle(londs[i] - del_lat/cos_factor[i], latds[i]) for i in 1:Npoints]

    # Perturbed North/South
    particles[3:4:end] .= [Particle(londs[i], latds[i] + del_lat) for i in 1:Npoints]
    particles[4:4:end] .= [Particle(londs[i], latds[i] - del_lat) for i in 1:Npoints]
end

function displacement_gradient_matrix_central(plonds, platds, dist_km)
    """
    Compute the displacement gradient matrix given particle positions at a fixed time
    Uses a central difference scheme to do this, with four points per grid cell

    Inputs:
        plonds: longitudes of particles
        platds: latitudes of particles
        dist_km: perturbation in position applied before starting simulation

    Outputs:
        B: (2,2,N) array where N is the number of grid points in the simulation.
        B[:,:,k] is the displacement gradient matrix for the k'th grid point. 
    """

    Ngpoints = length(plonds) ÷ 4 # Number of grid points

    dfac = Re / dist_km / 2

    cos_factor = cos.(deg2rad.(platds)) # Cos(latitude)

    # Derivative of x w.r.t. X
    xX = [deg2rad((plonds[4i-3] - plonds[4i-2]))*cos_factor[i]*dfac for i in 1:Ngpoints];
    # Derivative of y w.r.t. X 
    yX = [deg2rad((platds[4i-3] - platds[4i-2]))*dfac for i in 1:Ngpoints];
    # Derivative of x w.r.t. Y
    xY = [deg2rad((plonds[4i-1] - plonds[4i]))*cos_factor[i]*dfac for i in 1:Ngpoints];
    # Derivative of y w.r.t. Y 
    yY = [deg2rad((platds[4i-1] - platds[4i]))*dfac for i in 1:Ngpoints];

    # Displacement gradient matrix
    B = Array{eltype(xX)}(undef, 2, 2, Ngpoints)
    B[1,1,:] .= xX
    B[1,2,:] .= xY
    B[2,1,:] .= yX
    B[2,2,:] .= yY 
    # ^ Not the most elegant, but explicit 
    
    return B
end

function FTLE_over_grid(B, T)
    """
    Compute FTLE over a grid

    Inputs:
        B: displacement gradient matrix
        T: time after particle release which B corresponds to

    Outputs: 
        FTLE_grid: FTLE at each grid point
    """

    Ngpoints = size(B, 3) # Number of grid points

    FTLE_grid = Vector{Float64}(undef, Ngpoints) 

    for k in 1:Ngpoints
        # Gradient tensor
        Bk = B[:,:,k];
        # Right Cauchy-Green tensor
        CG = Bk' * Bk;
        # Largest eigenvalue
        lmax = maximum(eigvals(CG))
        # FTLE - in units of days^-1
        FTLE_grid[k] = FTLE_from_eigenvalue(lmax, T)
    end

    return FTLE_grid
end
