"""
Mixed Logit (MXL) Functions for Cycling Safety Choice Models

This module contains functions for estimating and simulating mixed logit models
with panel data, adapted from the reference implementation for the cycling safety project.
"""

import biogeme.biogeme as bio
import biogeme.database as db
from biogeme import models
from biogeme.expressions import (Beta, Variable, log, exp, 
                                 MonteCarlo, bioMultSum, bioDraws)
import numpy as np
import os
from pathlib import Path


def estimate_mxl(V, AV, CHOICE, obs_per_ind, num_draws, biodata_wide, model_name, output_dir=None):
    """
    Estimate a mixed logit model with panel data
    
    Args:
        V: List of dictionaries containing utility functions for each observation
        AV: Dictionary containing availability conditions  
        CHOICE: String name of choice variable
        obs_per_ind: Number of observations per individual
        num_draws: Number of Monte Carlo draws
        biodata_wide: Biogeme database in wide format
        model_name: Name of the model
        output_dir: Output directory for results (optional)
    
    Returns:
        Biogeme estimation results
    """
    # The conditional probability of the chosen alternative is a logit
    condProb = [models.loglogit(V[q], AV, Variable(f'{CHOICE}_{q}')) for q in range(obs_per_ind)] 

    # Take the product of the conditional probabilities
    condprobIndiv = exp(bioMultSum(condProb))   # exp to convert from logP to P again

    # The unconditional probability is obtained by simulation
    uncondProb = MonteCarlo(condprobIndiv)

    # The Log-likelihood is the log of the unconditional probability
    LL = log(uncondProb)

    # Create the Biogeme estimation object containing the data and the model
    biogeme = bio.BIOGEME(biodata_wide, LL, number_of_draws=num_draws)

    # Compute the null loglikelihood for reporting
    # Note that we need to compute it manually, as biogeme does not do this for panel data
    biogeme.nullLogLike = biodata_wide.get_sample_size() * obs_per_ind * np.log(1 / len(V[0]))

    # Set model name
    biogeme.modelName = model_name    
    
    # Configure output settings
    biogeme.generate_pickle = True
    biogeme.generate_html = True
    biogeme.save_iterations = False
    
    # Change to output directory if specified
    original_cwd = None
    if output_dir:
        original_cwd = os.getcwd()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

    try:
        # Estimate the parameters
        results = biogeme.estimate()
        
        # Save additional outputs if in output directory
        if output_dir:
            try:
                results.write_latex()
                results.write_pickle() 
                results.write_html()
            except AttributeError:
                results.writeLaTeX()
                results.writePickle()
                results.writeHTML()
                
    finally:
        # Always return to original directory
        if original_cwd:
            os.chdir(original_cwd)
    
    return results


def prepare_panel_data(dataframe, individual_id='RID', choice_var='CHOICE', features=None):
    """
    Prepare panel data for mixed logit estimation
    
    Args:
        dataframe: Pandas dataframe with panel data
        individual_id: Column name for individual identifier
        choice_var: Column name for choice variable
        features: (Optional) List of base feature names. If provided, the function will
                  attempt to subset the dataframe to only necessary columns.
    """
    # --- Data Cleaning ---
    # Biogeme requires a clean database with only numeric types and no NaNs.
    # We select only numeric columns to remove categorical/string columns.
    df_numeric = dataframe.select_dtypes(include=np.number)
    
    # Drop any remaining NaN values.
    df_clean = df_numeric.dropna()
    
    print(f"Dataframe cleaned for Biogeme. Shape before: {dataframe.shape}, After: {df_clean.shape}")
    if df_clean.shape[0] < dataframe.shape[0]:
        print(f"Dropped {dataframe.shape[0] - df_clean.shape[0]} rows due to NaN values.")

    # Create Biogeme database from the cleaned data
    biodata = db.Database('panel_data', df_clean)
    
    # Tell Biogeme which variable is the identifier of the individuals
    biodata.panel(individual_id)
    
    # Calculate the number of observations per individual
    obs_per_ind = df_clean[individual_id].value_counts().iloc[0]
    print(f'Number of observations per individual: {obs_per_ind}')
    
    # Create wide format database
    df_wide = biodata.generate_flat_panel_dataframe(identical_columns=None)
    
    # Rename the columns to run from columnname_{0} to columnname_{n}
    renumbered_columns = {}
    for col in df_wide.columns:
        parts = col.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            obs_index = int(parts[0]) - 1
            var_name = parts[1]
            renumbered_columns[col] = f'{var_name}_{obs_index}'

    df_wide.rename(columns=renumbered_columns, inplace=True)
    
    # Create Biogeme database object for wide format
    biodata_wide = db.Database('panel_data_wide', df_wide)
    
    print(f'Wide dataset shape: {df_wide.shape}')
    
    return biodata, biodata_wide, obs_per_ind


def create_random_parameters(obs_per_ind, distributions, num_draws_type='NORMAL_HALTON2'):
    """
    Create random parameters for mixed logit model
    
    Args:
        obs_per_ind: Number of observations per individual
        distributions: Dictionary specifying parameter distributions
                      Format: {'param_name': {'mean': Beta, 'sigma': Beta, 'dist': 'normal|lognormal', 'sign': 1|-1}}
        num_draws_type: Type of draws for Monte Carlo integration
        
    Returns:
        Tuple of (random_parameters, variables_dict)
    """
    random_params = {}
    variables_dict = {}
    
    for param_name, config in distributions.items():
        mean_param = config['mean']
        sigma_param = config['sigma'] 
        dist_type = config.get('dist', 'normal')
        sign = config.get('sign', 1)
        
        # Create random draws
        draw_name = f'{param_name}_rnd'
        draws = bioDraws(draw_name, num_draws_type)
        
        # Create random parameter based on distribution type
        if dist_type == 'normal':
            random_param = mean_param + sigma_param * draws
        elif dist_type == 'lognormal':
            if sign == -1:
                random_param = -exp(mean_param + sigma_param * draws)
            else:
                random_param = exp(mean_param + sigma_param * draws)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        random_params[param_name] = random_param
        
        # Create variables for all observations
        variables_dict[param_name] = {
            q: Variable(f'{param_name}_{q}') for q in range(obs_per_ind)
        }
    
    return random_params, variables_dict


def create_utility_functions(random_params, variables_dict, obs_per_ind, alternatives=[1, 2]):
    """
    Create utility functions for mixed logit model
    
    Args:
        random_params: Dictionary of random parameters
        variables_dict: Dictionary of variables for each observation
        obs_per_ind: Number of observations per individual
        alternatives: List of alternative numbers
        
    Returns:
        List of utility function dictionaries for each observation
    """
    V = []
    
    for q in range(obs_per_ind):
        V_q = {}
        
        for alt in alternatives:
            utility_components = []
            
            # Add utility components based on available parameters and variables
            for param_name, random_param in random_params.items():
                if param_name in variables_dict:
                    var_name = f'{param_name}{alt}_{q}' if f'{param_name}{alt}' in variables_dict[param_name][q].name else f'{param_name}_{alt}_{q}'
                    if var_name in [v.name for v in variables_dict[param_name].values()]:
                        variable = variables_dict[param_name][q]
                        utility_components.append(random_param * variable)
            
            V_q[alt] = sum(utility_components) if utility_components else 0
            
        V.append(V_q)
    
    return V


def apply_data_cleaning(dataframe, individual_id='RID', min_obs=15, drop_problematic_rid=True):
    """
    Apply data cleaning steps from reference implementation

    Args:
        dataframe: Input dataframe
        individual_id: Column name for individual identifier
        min_obs: Minimum number of observations per individual
        drop_problematic_rid: Whether to drop the problematic rows from RID 63
            (co-author flagged the last 15 rows as duplicated + stray SEQ row)

    Returns:
        Cleaned dataframe
    """
    df_clean = dataframe.copy()

    print(f"Original data shape: {df_clean.shape}")

    # Drop the last 15 rows of RID 63 (duplicated rows + stray SEQ row, per co-author)
    if drop_problematic_rid and 63 in df_clean[individual_id].values:
        print("Dropping last 15 rows of RID 63 (duplicated + stray SEQ)...")
        mask = df_clean[individual_id] == 63
        last_15_idx = df_clean[mask].tail(15).index
        df_clean = df_clean.drop(index=last_15_idx)
        print(f"Dropped {len(last_15_idx)} rows from RID 63")

    # Drop RIDs with less than minimum observations
    print(f'Dropping RIDs with less than {min_obs} observations...')
    rid_counts = df_clean[individual_id].value_counts()
    rids_to_drop = rid_counts[rid_counts < min_obs].index

    print(f'Dropping {len(rids_to_drop)} RIDs with less than {min_obs} observations')
    df_clean = df_clean[~df_clean[individual_id].isin(rids_to_drop)]

    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Unique RIDs: {df_clean[individual_id].nunique()}")

    if drop_problematic_rid:
        assert df_clean.shape[0] == 11190, f"Expected 11190 rows, got {df_clean.shape[0]}"
        assert df_clean[individual_id].nunique() == 746, (
            f"Expected 746 individuals, got {df_clean[individual_id].nunique()}"
        )

    return df_clean


def extract_mxl_metrics(results, obs_per_ind, n_individuals):
    """
    Extract key metrics from MXL estimation results
    
    Args:
        results: Biogeme estimation results
        obs_per_ind: Number of observations per individual  
        n_individuals: Number of individuals
        
    Returns:
        Dictionary with model metrics
    """
    # Handle different result object structures
    if hasattr(results, 'data'):
        data = results.data
    else:
        data = results
    
    # Calculate basic metrics
    log_like = data.logLike
    n_params = len(data.betaValues)
    n_obs = n_individuals * obs_per_ind
    
    # Calculate AIC and BIC
    aic = 2 * n_params - 2 * log_like
    bic = np.log(n_individuals) * n_params - 2 * log_like
    
    # Get pseudo R-squared
    pseudo_r2 = data.rhoSquare
    
    return {
        'log_likelihood': log_like,
        'n_parameters': n_params,
        'n_observations': n_obs,
        'n_individuals': n_individuals,
        'obs_per_individual': obs_per_ind,
        'AIC': aic,
        'BIC': bic,
        'pseudo_r2': pseudo_r2
    }


def estimate_mnl(V, AV, CHOICE, biodata, model_name, output_dir=None):
    """
    Estimate a multinomial logit model (without panel structure)
    
    Args:
        V: Dictionary containing utility functions {alt: utility}
        AV: Dictionary containing availability conditions  
        CHOICE: String name of choice variable
        biodata: Biogeme database
        model_name: Name of the model
        output_dir: Output directory for results (optional)
    
    Returns:
        Biogeme estimation results
    """
    # Calculate the probability using logit model
    prob = models.logit(V, AV, Variable(CHOICE))
    
    # Log-likelihood function
    LL = log(prob)
    
    # Create the Biogeme estimation object
    biogeme = bio.BIOGEME(biodata, LL)
    
    # Compute the null loglikelihood for reporting
    biogeme.nullLogLike = biodata.get_sample_size() * np.log(1 / len(V))
    
    # Set model name
    biogeme.modelName = model_name    
    
    # Configure output settings
    biogeme.generate_pickle = True
    biogeme.generate_html = True
    biogeme.save_iterations = False
    
    # Change to output directory if specified
    original_cwd = None
    if output_dir:
        original_cwd = os.getcwd()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

    try:
        # Estimate the parameters
        results = biogeme.estimate()
        
        # Save additional outputs if in output directory
        if output_dir:
            try:
                results.write_latex()
                results.write_pickle() 
                results.write_html()
            except AttributeError:
                results.writeLaTeX()
                results.writePickle()
                results.writeHTML()
                
    finally:
        # Always return to original directory
        if original_cwd:
            os.chdir(original_cwd)
    
    return results


def estimate_wtp_mnl(V, AV, CHOICE, biodata, model_name, output_dir=None):
    """
    Estimate a multinomial logit model in WTP space (without panel structure)
    
    Args:
        V: Dictionary containing utility functions {alt: utility}
        AV: Dictionary containing availability conditions  
        CHOICE: String name of choice variable
        biodata: Biogeme database
        model_name: Name of the model
        output_dir: Output directory for results (optional)
    
    Returns:
        Biogeme estimation results
    """
    # Calculate the probability using logit model
    prob = models.logit(V, AV, Variable(CHOICE))
    
    # Log-likelihood function
    LL = log(prob)
    
    # Create the Biogeme estimation object
    biogeme = bio.BIOGEME(biodata, LL)
    
    # Compute the null loglikelihood for reporting
    biogeme.nullLogLike = biodata.get_sample_size() * np.log(1 / len(V))
    
    # Set model name
    biogeme.modelName = model_name    
    
    # Configure output settings
    biogeme.generate_pickle = True
    biogeme.generate_html = True
    biogeme.save_iterations = False
    
    # Change to output directory if specified
    original_cwd = None
    if output_dir:
        original_cwd = os.getcwd()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

    try:
        # Estimate the parameters
        results = biogeme.estimate()
        
        # Save additional outputs if in output directory
        if output_dir:
            try:
                results.write_latex()
                results.write_pickle() 
                results.write_html()
            except AttributeError:
                results.writeLaTeX()
                results.writePickle()
                results.writeHTML()
                
    finally:
        # Always return to original directory
        if original_cwd:
            os.chdir(original_cwd)
    
    return results


def estimate_wtp_mxl(V, AV, CHOICE, obs_per_ind, num_draws, biodata_wide, model_name, output_dir=None):
    """
    Estimate a mixed logit model in WTP space with panel data
    
    Args:
        V: List of dictionaries containing utility functions for each observation
        AV: Dictionary containing availability conditions  
        CHOICE: String name of choice variable
        obs_per_ind: Number of observations per individual
        num_draws: Number of Monte Carlo draws
        biodata_wide: Biogeme database in wide format
        model_name: Name of the model
        output_dir: Output directory for results (optional)
    
    Returns:
        Biogeme estimation results
    """
    # The conditional probability of the chosen alternative is a logit
    condProb = [models.loglogit(V[q], AV, Variable(f'{CHOICE}_{q}')) for q in range(obs_per_ind)] 

    # Take the product of the conditional probabilities
    condprobIndiv = exp(bioMultSum(condProb))   # exp to convert from logP to P again

    # The unconditional probability is obtained by simulation
    uncondProb = MonteCarlo(condprobIndiv)

    # The Log-likelihood is the log of the unconditional probability
    LL = log(uncondProb)

    # Create the Biogeme estimation object containing the data and the model
    biogeme = bio.BIOGEME(biodata_wide, LL, number_of_draws=num_draws)

    # Compute the null loglikelihood for reporting
    # Note that we need to compute it manually, as biogeme does not do this for panel data
    biogeme.nullLogLike = biodata_wide.get_sample_size() * obs_per_ind * np.log(1 / len(V[0]))

    # Set model name
    biogeme.modelName = model_name    
    
    # Configure output settings
    biogeme.generate_pickle = True
    biogeme.generate_html = True
    biogeme.save_iterations = False
    
    # Change to output directory if specified
    original_cwd = None
    if output_dir:
        original_cwd = os.getcwd()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

    try:
        # Estimate the parameters
        results = biogeme.estimate()
        
        # Save additional outputs if in output directory
        if output_dir:
            try:
                results.write_latex()
                results.write_pickle() 
                results.write_html()
            except AttributeError:
                results.writeLaTeX()
                results.writePickle()
                results.writeHTML()
                
    finally:
        # Always return to original directory
        if original_cwd:
            os.chdir(original_cwd)
    
    return results


def simulate_mxl(V, AV, CHOICE, obs_per_ind, num_draws, biodata_wide, betas, model_name):
    """
    Simulate a panel mixed-logit model with fixed parameters and compute fit measures.
    
    Args:
        V: List of dictionaries containing utility functions for each observation
        AV: Dictionary containing availability indicators
        CHOICE: String name of choice variable
        obs_per_ind: Number of observations per individual
        num_draws: Number of draws for simulation
        biodata_wide: Biogeme database in wide format
        betas: Dictionary of parameter estimates {param_name: value}
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with simulation results: LL, rho_square, Pchosen_seq
    """
    # The conditional probability of the chosen alternative is a logit
    condProb = [models.loglogit(V[q], AV, Variable(f'{CHOICE}_{q}')) for q in range(obs_per_ind)] 

    # Take the product of the conditional probabilities
    condprobIndiv = exp(bioMultSum(condProb))   # exp to convert from logP to P again

    # The unconditional probability is obtained by simulation
    uncondProb = MonteCarlo(condprobIndiv)

    # The Log-likelihood is the log of the unconditional probability
    LL = log(uncondProb)

    # Simulate the model on the test database with fixed parameters
    simulated_loglike = LL.get_value_c(
        database=biodata_wide,
        betas=betas,
        number_of_draws=num_draws,
        aggregation=False,
        prepare_ids=True,
    )
    LLsim = simulated_loglike.sum()

    # Rho-square calculation
    LLnull = biodata_wide.get_sample_size() * obs_per_ind * np.log(1 / len(V[0]))
    rho_square = 1 - (LLsim / LLnull)

    # Add the results to a dictionary
    simresults = {
        "Pchosen_seq": simulated_loglike, 
        "LL": LLsim, 
        "rho_square": rho_square,
        "model_name": model_name
    }
    return simresults


def print_mxl_results(results):
    """Print formatted results for MXL model"""
    print("\n" + "="*60)
    print("MIXED LOGIT MODEL ESTIMATION RESULTS")
    print("="*60)
    
    # Handle different result object structures
    if hasattr(results, 'data'):
        data = results.data
    else:
        data = results
    
    # Model fit statistics
    print(f"Log-likelihood: {data.logLike:.6f}")
    print(f"Rho-square: {data.rhoSquare:.6f}")
    print(f"Number of estimated parameters: {len(data.betaValues)}")
    print(f"Number of observations: {data.numberOfObservations}")
    
    # Parameter estimates
    print("\nParameter Estimates:")
    print("-" * 60)
    
    try:
        param_estimates = data.get_estimated_parameters()
        print(param_estimates.to_string())
    except Exception:
        # Fallback if get_estimated_parameters() is not available
        print("Beta values:")
        for param, value in zip(data.betaNames, data.betaValues):
            print(f"  {param}: {value:.6f}")