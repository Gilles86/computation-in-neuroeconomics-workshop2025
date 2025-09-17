## The following code implements adaptive effciient coding according to an encoding prior

import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize
import pandas as pd

import probTools as tools
probability_scale = tools.probability_scale

#  Encoded distribution for an input quantity - alwys happens on encoding_scale
# Does the internal tranformation and then adds internal noise as well.
def MI_efficient_encoding(input_scale, p_input, sigma_rep, encoding_type):
    # rep_scale always from 0 to 1 (full capacity) but number of points in transform based on input grid
    rep_scale = np.linspace(0.0, 1.0, len(input_scale)) 
    # rep_scale is being defined in every finction since the vectorized implementation nedds to the rep_scale to be the same leangth as input_scale to work
    # C is limited - we take it as 1 here
    
    # Add sensory noise for each value of p_input - encoding on encoding_scale with inputs 
    p_m_given_p_input = tools.truncated_noise(tools.cdf_adaptive_encoding(p_input, encoding_type, input_scale), sigma_rep=sigma_rep,
                                            grid=rep_scale)
    return p_m_given_p_input

# Given that a noisy encoding of quantity was ensued, prob_estimates gives exact decoded point estimates from subjects based on bayesian inference and loss function
def subject_prob_estimate(input_scale, sigma_rep, encoding_type, decoding_type, loss_exp=2):
    # rep_scale always from 0 to 1 (full normalized capacity) but number of points in transform based on encoding grid
    rep_scale = np.linspace(0.0, 1.0, len(input_scale))

    ## This one is used by subject for their bayesian decoded distribution
    p_m_given_p_scale = tools.truncated_noise(tools.cdf_adaptive_encoding(input_scale, encoding_type, input_scale), sigma_rep=sigma_rep,
                                    grid=rep_scale)

    # encoding_scale x m (subject's bayesian decode)
    p_OutScale_given_m = p_m_given_p_scale * tools.decoding_prior(input_scale, decoding_type)[:, np.newaxis]
    p_OutScale_given_m = p_OutScale_given_m / trapezoid(p_OutScale_given_m, input_scale, axis=0)[np.newaxis, :]

    # Mean of output distribution (0th axis)
    x0 = trapezoid(input_scale[:, np.newaxis]*p_OutScale_given_m, input_scale, axis=0)
    if loss_exp == 2:
        prob_estimates = x0
    else:
        prob_estimates = []
        for ix in range(len(x0)):
            # Define the cost function for non-circular variables
            cost_function = lambda probest: np.sum(p_OutScale_given_m[:, ix] * (input_scale - probest)**(loss_exp/2))
            
            # Define the Jacobian (gradient) for non-circular variables
            jacobian = lambda probest: -np.sum(p_OutScale_given_m[:, ix] * (.5 * loss_exp * (input_scale - probest)**(loss_exp/2 - 1)))
            
            # Perform the minimization
            x = minimize(cost_function, x0[ix], method='BFGS', jac=jacobian).x[0]
            prob_estimates.append(x)
        prob_estimates = np.array(prob_estimates)
    return prob_estimates

# Given that a noisy encoding of stimulus was ensued and given that prob_estimates gives exact points where 
# each point in the bayesian observer's brain ends up - this gives distribution of estimates
def output_prob_distribution(input_scale, p_input, sigma_rep, encoding_type, decoding_type, loss_exp = 2):
    rep_scale = np.linspace(0.0, 1.0, len(input_scale))
    p_m_given_p_input = MI_efficient_encoding(input_scale, p_input, sigma_rep, encoding_type)
    prob_estimates = subject_prob_estimate(input_scale, sigma_rep, encoding_type, decoding_type, loss_exp=loss_exp)
    output_scale, output_prob_dist = tools.prob_transform(input_scale, prob_estimates, p_m_given_p_input)
    return output_scale, output_prob_dist 

# Given mean of distribution of estimates
def output_mean(input_scale, p_input, sigma_rep, encoding_type, decoding_type, loss_exp = 2):
    output_scale, output_prob_dist  = output_prob_distribution(input_scale, p_input, sigma_rep, encoding_type, decoding_type, loss_exp = loss_exp)
    mean_output = trapezoid(output_scale[np.newaxis, :]*output_prob_dist, output_scale, axis=1) 
    return mean_output

# gives varianvce of distribution of estimates
def output_variance(input_scale, p_input, sigma_rep, encoding_type, decoding_type, loss_exp = 2):
    output_scale, output_prob_dist  = output_prob_distribution(input_scale, p_input, sigma_rep, encoding_type, decoding_type, loss_exp = loss_exp)
    mean_output = output_mean(input_scale, p_input, sigma_rep, encoding_type, decoding_type, loss_exp = 2)
    output_variances = np.sum((output_prob_dist * (output_scale[np.newaxis ,:] - mean_output[:, np.newaxis])**2), axis = 1)
    return output_variances

# Calculates the rolling mean of the variability across the range
def output_variability(input_scale, p_input, sigma_rep, encoding_type, decoding_type, loss_exp=2, num_subjects=71, num_samples=2, rolling_size=5):
    all_distances_from_mean = np.zeros((num_subjects, len(p_input)))
    output_scale, output_prob_dist = output_prob_distribution(input_scale, p_input, sigma_rep, encoding_type, decoding_type, loss_exp = loss_exp)

    for j in range(num_subjects):
        distances_from_mean = np.zeros(len(p_input))
        for i in range(len(p_input)):
            # Sample two points randomly from the distribution    
            sampled_indices = np.random.choice(np.arange(len(output_scale)), size=num_samples, p=output_prob_dist[i] / np.trapz(output_prob_dist[i]))
            sampled_points = output_scale[sampled_indices]

            # Calculate the mean of the sampled points
            mean_sampled_points = np.mean(sampled_points)

            # Calculate the sum of the absolute distances from the mean
            distances_from_mean[i] = np.mean(np.abs(sampled_points - mean_sampled_points))

        all_distances_from_mean[j] = distances_from_mean

    avg_distance_from_mean = np.mean(all_distances_from_mean, axis=0)

    # Calculate rolling mean
    rolling_variability = pd.Series(avg_distance_from_mean).rolling(window=rolling_size, min_periods=5).mean().to_numpy()
    
    return rolling_variability
