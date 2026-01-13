using StaticArrays

Re = 6.371e6 # Average Earth radius in meters

function FTLE_grad(FTLE_grid)
    """
    Compute derivative of FTLE field map

    Inputs:
        FTLE field map as given by FTLE_over_grid function

    Outputs:
        Gradient of Field map dσdx and dσdy both lists of length number of grid points
    """
    dσdx, dσdy = ∇(Field(FTLE_grid, grid), radius = Re)
    return dσdx, dσdy
end

function FTLE_hessian(FTLE_grid)
    """
    Compute the Hessian over the FTLE field map.

    Inputs:
        FTLE field map as given by FTLE_over_grid function

    Outputs:
        Hessian at each point on the field map as a 2x2xN array (N is number of grid points in grid space)
    """
    dσdx, dσdy = FTLE_grad(FTLE_grid)
    H11, H12 = ∇(dσdx, radius = Re)
    H21, H22 = ∇(dσdy, radius = Re)
    
    Npoints = length(dσdx)

    H = Array{eltype(dσdx)}(undef, 2, 2, Npoints)
    H[1,1,:] .= H11
    H[1,2,:] .= H12
    H[2,1,:] .= H21
    H[2,2,:] .= H22

    return H
end

function filter_FTLE(FTLE_grid)
    """
    Filter FTLE grid for points that are located on maximising ridges
    Conditions: 
        Gradient orthogonal to minimum eigenvalue of hessian
        Minimum eigenvalue of hessian is less than zero
        FTLE value is greater than 50% of the global maximum

    Inputs:
        FTLE field map as computed by FTLE_over_grid
    
    Outputs:
        Filtered list of FTLE field map containing only those points located on maximising ridges
    """
    dσdx, dσdy = FTLE_grad(FTLE_grid)
    H = FTLE_hessian(FTLE_grid)
    Npoints, orth_tol, FTLE_tol = length(dσdx), 0.001, 0.5*maximum(FTLE_grid)
    
    filter_list = fill(NaN, Npoints)

    for i in 1:Npoints  
        Hk, grad = eigen(Symmetric(H[:, :, i])), SVector(dσdx[i], dσdy[i])
        λ = minimum(Hk.values)
        v = Hk.vectors[:,1]
        orth = dot(grad, v)
        if λ < 0 && abs(orth) < orth_tol && FTLE_grid[i] > FTLE_tol
            filter_list[i] = FTLE_grid[i]
        end
    end

    return filter_list
end