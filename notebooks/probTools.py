import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

grid_points = 200
input_scale_full = np.linspace(0, 1, grid_points)
input_scale_half1 = np.linspace(0, 0.5, int(grid_points))
input_scale_half2 = np.linspace(0.5, 1, int(grid_points))

input_scale_low = np.linspace(0, 0.33, 250)
input_scale_mid = np.linspace(0.33, 0.65, 250)
input_scale_high = np.linspace(0.66, 1, 250)

# Natural scale for probabilities
probability_scale = np.linspace(0, 1, grid_points)

# Encoding prior here dictates the encoding function through efficient coding - defined on the full probability scale
# Specifying an encoding prior is equivalent to specifying an encoding function under efficient coding
def encoding_prior(encoding_type):
    if encoding_type == "uniform":
        encoding_prior = np.repeat(1, len(probability_scale)) / len(probability_scale)  # uniform
    elif encoding_type == "u":
        # U-shaped prior: higher probability at extremes (0 and 1) - more extreme
        x = np.linspace(0, 1, len(probability_scale))
        encoding_prior = x**4 + (1-x)**4  # sharper U-shape
        encoding_prior = encoding_prior / np.sum(encoding_prior)  # normalize
    elif encoding_type == "hump":
        # Hump-shaped prior: higher probability in the middle (around 0.5) - less extreme
        x = np.linspace(0, 1, len(probability_scale))
        encoding_prior = 2 * x * (1 - x) + 0.3  # gentler hump with baseline
        encoding_prior = encoding_prior / np.sum(encoding_prior)  # normalize
    elif encoding_type == "skewed_u":
        # Skewed U-shaped prior: asymmetric U with bias toward one extreme
        x = np.linspace(0, 1, len(probability_scale))
        encoding_prior = x**2 + 2*(1-x)**2 + 0.7 # skewed toward 0
        encoding_prior = encoding_prior / np.sum(encoding_prior)  # normalize
    elif encoding_type == "skewed_hump":
        # Skewed hump-shaped prior: asymmetric hump with bias - less extreme
        x = np.linspace(0, 1, len(probability_scale))
        encoding_prior = 2 * x**2 * (1 - x) + 0.5  # gentler skewed hump with baseline
        encoding_prior = encoding_prior / np.sum(encoding_prior)  # normalize
    return encoding_prior

# Encoding function depending on the cdf of the encoding prior. The cdf is taken until the input scale (not always the full probability scale)
# so the shape is preserved on the whole scale and the part correspodning to the input is taken and normalized. This is ebecause encoding is assumed adaptive on some prior of probabilities in the world
def cdf_adaptive_encoding(x, encoding_type, input_scale=None): # goes from 0 to 1
    x = np.asarray(x)
    prior = encoding_prior(encoding_type)
    
    # If input_scale is provided, extract the corresponding portion from probability_scale
    if input_scale is not None:
        input_scale = np.asarray(input_scale)
        # Find indices in probability_scale that correspond to the input_scale range
        min_input = input_scale[0]
        max_input = input_scale[-1]
        
        # Find the mask for probability_scale that falls within input_scale range
        mask = (probability_scale >= min_input) & (probability_scale <= max_input)
        
        # Extract the prior values for this range and normalize
        prior_subset = prior[mask]
        prior_subset = prior_subset / np.sum(prior_subset)  # normalize
        
        cdf_whole = np.cumsum(prior_subset)
        scale_to_use = probability_scale[mask]
    else:
        cdf_whole = np.cumsum(prior)
        scale_to_use = probability_scale
    
    # Check if x has a single element or multiple elements
    if x.size == 1:
        cdf_values = cdf_whole[scale_to_use <= x][-1] if np.any(scale_to_use <= x) else 0.0
    else:
        cdf_values = [cdf_whole[scale_to_use <= xi][-1] if np.any(scale_to_use <= xi) else 0.0 for xi in x]
    return np.asarray(cdf_values)

# Encoding function that transforms the input scale to the encoding scale - no efficient coding - just direcvtly defined encoding - this is when looking at the bayesian decoding origin
def encoding_function(p_input, encoding_type):
    if encoding_type is "linear":
        encoding = p_input
    elif encoding_type is "concave":
        # Concave encoding: more weight on lower values
        encoding = np.log(2 * p_input + 1)/np.log(3)
    elif encoding_type is "convex":
        # Convex encoding: more weight on higher values
        encoding = (3**p_input-1)/2
    elif encoding_type is "cave_vex":
        encoding = 2*p_input**3  - 3*p_input**2 + 2*p_input
    elif encoding_type is "vex_cave":
        encoding = 3*p_input**2 - 2*p_input**3
    return encoding

# Bounded Decoding prior is bounded on the input - here dictates the bayesian inference part only.
def decoding_prior(input_scale, decoding_type):
    if decoding_type == "uniform":
        prior = np.ones(len(input_scale), dtype=float)
        prior = prior/np.sum(prior)
    elif decoding_type == "u":
        # U-shaped prior: higher probability at extremes (0 and 1)
        x = np.linspace(input_scale[0], input_scale[-1], len(input_scale))
        prior = x**2 + (1-x)**2  # quadratic U-shape
        prior = prior / np.sum(prior)  # normalize
    elif decoding_type == "hump":
        # Hump-shaped prior: higher probability in the middle (around 0.5)
        x = np.linspace(input_scale[0], input_scale[-1], len(input_scale))
        prior = 4 * x * (1 - x)  # inverted U-shape (beta(2,2) like)
        prior = prior / np.sum(prior)  # normalize
    elif decoding_type == "skewed_u":
        # Skewed U-shaped prior: asymmetric U with bias toward one extreme
        x = np.linspace(input_scale[0], input_scale[-1], len(input_scale))
        prior = x**3 + 2*(1-x)**2  # skewed toward 0
        prior = prior / np.sum(prior)  # normalize
    elif decoding_type == "skewed_hump":
        # Skewed hump-shaped prior: asymmetric hump with bias
        x = np.linspace(input_scale[0], input_scale[-1], len(input_scale))
        prior = 6 * x**2 * (1 - x)  # skewed hump toward higher values
        prior = prior / np.sum(prior)  # normalize
    return prior

    
# This is the noisy encoding that the subject has under bounded resources. We use this when simulating the efficient coding version.
# The grid reflects the encoding scale whgere the truncation happens.
def truncated_noise(m, sigma_rep, grid):
    # Ensure that m and sigma_rep are NumPy arrays
    m = np.atleast_1d(m)
    sigma_rep = np.atleast_1d(sigma_rep)

    if isinstance(sigma_rep, (int, float)):  # Check if sigma_rep is a scalar
        # If sigma_rep is a scalar, use it for all points in m
        sigma_rep = np.full_like(m, sigma_rep)

    # Ensure that grid is a NumPy array
    grid = np.array(grid)

    # Calculate the truncated normal distribution for each pair of mean and standard deviation
    truncBoth = ss.truncnorm.pdf(grid,
        (grid[..., 0] - m[:, np.newaxis]) / sigma_rep[:, np.newaxis],
        (grid[..., -1] - m[:, np.newaxis]) / sigma_rep[:, np.newaxis],
        m[:, np.newaxis],
        sigma_rep[:, np.newaxis]
    )
    return truncBoth

# This is the Gaussian noise function without truncation for comparison or when bounds are not needed
def gaussian_noise(m, sigma_rep, grid):
    # Ensure that m and sigma_rep are NumPy arrays
    m = np.atleast_1d(m)
    sigma_rep = np.atleast_1d(sigma_rep)

    if isinstance(sigma_rep, (int, float)):  # Check if sigma_rep is a scalar
        # If sigma_rep is a scalar, use it for all points in m
        sigma_rep = np.full_like(m, sigma_rep)

    # Ensure that grid is a NumPy array
    grid = np.array(grid)

    # Calculate the normal distribution for each pair of mean and standard deviation
    gaussian = ss.norm.pdf(grid, m[:, np.newaxis], sigma_rep[:, np.newaxis])
    return gaussian


# Transform probability distribution over a grid of random variable to another random variable grid
# and then extrapolate outside the new grid with 0.
def prob_transform(grid, new_grid, p, bins=101, interpolation_kind='linear'):
    grid = np.array(grid)
    
    # For every bin in x_stim, calculate the probability mass within that bin
    dx = grid[..., 1:] - grid[..., :-1]
    p_mass = ((p[..., 1:] + p[..., :-1]) / 2) * dx

    if any(np.diff(new_grid)<=0): # For non monotonic transforms use histogram
        # Get the center of every bin
        x_value = new_grid[:-1] + dx / 2.
        ps = []
        for ix in range(len(p)):
            h, edges = np.histogram(x_value, bins=bins, weights=p_mass[ix], density=True)
            ps.append(h)

        ps = np.array(ps)
        new_grid = (edges[1:] + edges[:-1]) / 2

    else: #use the monotonic transformation formula analytic one
        ps = p
        ps[...,:] = ps[...,:]/abs(np.gradient(new_grid, grid))

    # ps_new = np.zeros(ps.shape)
    # We asssume here that the subject can never value any option outside of the val_estimates range
    # due to perceptual effects on the grid of values. 
    # new_grid_n = np.concatenate(([np.min(new_grid)-1e-6], new_grid, [np.max(new_grid)+1e-6]), axis=0)
    # ps_new = np.concatenate((np.zeros((len(ps), 1)), ps, np.zeros((len(ps), 1))), axis=1)

    f = interpolate.interp1d(new_grid, ps, axis=1,
                                 bounds_error=False, kind=interpolation_kind, fill_value=0.0)
    ps = f(grid)
    ps /= abs(trapezoid(ps, grid, axis=1)[:, np.newaxis])

    return grid, ps

