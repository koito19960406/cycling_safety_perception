# Biogeme
from __future__ import annotations
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import log, exp, PanelLikelihoodTrajectory, Variable, MonteCarlo, exp, bioMultSum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid


def estimate_mnl(V, AV, CHOICE, database, model_name):
    '''
    Function to estimate the MNL model

    Parameters:
    V: a dictionary containing the utility functions for each alternative
    AV: a dictionary containing availability conditions
    CHOICE: an integer containing the choice variable
    database: database object
    model_name: name of the model

    Returns:
    results: estimation results
    '''

    # Define the choice model: The function models.logit() computes the MNL choice probabilities of the chosen alternative given the V. 
    prob = models.logit(V, AV, CHOICE)

    # Define the log-likelihood by taking the log of the choice probabilities of the chosen alternative
    LL = log(prob)
   
    # Create the Biogeme object containing the object database and the formula for the contribution to the log-likelihood of each row using the following syntax:
    biogeme = bio.BIOGEME(database, LL)
    
    # The following syntax passes the name of the model:
    biogeme.modelName = model_name

    # Some object settings regaridng whether to save the results and outputs 
    biogeme.generate_pickle = False
    biogeme.generate_html = False
    biogeme.save_iterations = False

    # Syntax to calculate the null log-likelihood. The null-log-likelihood is used to compute the rho-square 
    biogeme.calculate_null_loglikelihood(AV)

    # This line starts the estimation and returns the results object.
    results = biogeme.estimate()
    return results

def simulate_mnl(V, AV, CHOICE, database, betas, model_name):

    # Simulate the model on the test database
    simulate = {'P1': exp(V[1]) / (exp(V[1]) + exp(V[2])),
                'P2': exp(V[2]) / (exp(V[1]) + exp(V[2])),
                'Pchosen': (CHOICE == 1) * (exp(V[1]) / (exp(V[1]) + exp(V[2]))) + (CHOICE == 2) * (exp(V[2]) / (exp(V[1]) + exp(V[2]))),
                'V1': V[1],
                'V2': V[2],
                }

    # Run the simulation
    biogeme_sim = bio.BIOGEME(database, simulate)   
    Psim = biogeme_sim.simulate(betas)

    # Compute the log-likelihood of the simulated data
    LLsim = np.log(Psim['Pchosen']).sum()

    LLnull = biogeme_sim.calculate_null_loglikelihood(AV)

    rho_square = 1 - (LLsim / LLnull)

    # Add the results to a dictionary
    simresults ={"V1": Psim['V1'], "V2": Psim['V2'],"P1": Psim['P1'], "P2": Psim['P2'],"Pchosen": Psim['Pchosen'], "LL": LLsim, "rho_square": rho_square,"model_name": model_name}
    
    return simresults


def estimate_mxl(V,AV,CHOICE,obs_per_ind,num_draws,biodata_wide,model_name):


    # The conditional probability of the chosen alternative is a logit
    condProb = [models.loglogit(V[q], AV, Variable(f'{CHOICE}_{q}')) for q in range(obs_per_ind)] 

    # Take the product of the conditional probabilities
    condprobIndiv = exp(bioMultSum(condProb))   # exp to convert from logP to P again

    # The unconditional probability is obtained by simulation
    uncondProb = MonteCarlo(condprobIndiv)

    # The Log-likelihood is the log of the unconditional probability
    LL = log(uncondProb)

    # Create the Biogeme estimation object containing the data and the model
    num_draws = num_draws
    biogeme = bio.BIOGEME(biodata_wide , LL, number_of_draws=num_draws)

    # Compute the null loglikelihood for reporting
    # Note that we need to compute it manually, as biogeme does not do this for panel data
    biogeme.nullLogLike = biodata_wide.get_sample_size() * obs_per_ind * np.log(1 / len(V[0]))

    # Set reporting levels
    biogeme.generate_pickle = False
    biogeme.generate_html = False
    biogeme.save_iterations = False
    biogeme.modelName = model_name    

    # Estimate the parameters and print the results
    results = biogeme.estimate()
    return results

def simulate_mxl(
    V,                 # dict of utility functions, {alt: expression}
    AV,                # dict of availability indicators
    CHOICE,            # name (str) or Variable object of the choice column
    obs_per_ind,       # number of observations per individual
    num_draws,         # number of draws for the simulation
    biodata_wide,      # biogeme.database.Database in *long* form
    betas,             # dict of parameter estimates
    model_name         # name of the model for reporting
    ):
    """Simulate a panel mixed-logit model and compute fit measures."""

    # --- 1. Expressions -----------------------------------------------------
    # The conditional probability of the chosen alternative is a logit
    condProb = [models.loglogit(V[q], AV, Variable(f'{CHOICE}_{q}')) for q in range(obs_per_ind)] 

    # Take the product of the conditional probabilities
    condprobIndiv = exp(bioMultSum(condProb))   # exp to convert from logP to P again

    # The unconditional probability is obtained by simulation
    uncondProb = MonteCarlo(condprobIndiv)

    # The Log-likelihood is the log of the unconditional probability
    LL = log(uncondProb)

    # Simulate the model on the test database
    simulated_loglike = LL.get_value_c(
        database=biodata_wide,
        betas=betas,
        number_of_draws = num_draws,
        aggregation=False,
        prepare_ids=True,
    )
    LLsim = simulated_loglike.sum()

    # Rho-square calculation
    LLnull = biodata_wide.get_sample_size() * obs_per_ind * np.log(1 / len(V[0]))
    rho_square = 1 - (LLsim / LLnull)

    # Add the results to a dictionary
    simresults ={"Pchosen_seq": simulated_loglike, "LL": LLsim, "rho_square": rho_square,"model_name": model_name}
    return simresults

def estimate_LC(V, AV, nu, CHOICE, database,model_name):
    
    '''
    Function to estimate the LC models

    Parameters:
    V: a list of dictionaries containing the utility functions for each class
    AV: a dictionary containing availability conditions
    nu is a list of value function for the class membership model
    CHOICE: choice variable
    database: database object
    model_name: name of the model

    Returns:
    results: estimation results
    '''
    # Determine the number of classes
    n_classes = len(V)

    # Compute the probabilities of the chosen alternative conditional the class
    prob = []
    for i in range(n_classes):
        prob.append(models.logit(V[i], AV, CHOICE))
        
    # Compute likelihood of the sequence of choices for each individual, conditional on the class
    Pseq = []
    for i in range(n_classes):
        Pseq.append(PanelLikelihoodTrajectory(prob[i]))

    # Compute class membership probabilities for each individual using the value function nu 
    P_class = {k: models.logit({j: nu[j] for j in range(n_classes)}, None, k) for k in range(n_classes)}
    
    # Compute the unconditional likelihood of the sequence of choices for each individual
    Prob_indiv = bio.bioMultSum([P_class[i] * Pseq[i] for i in range(n_classes)])
    
    # Take the log of the likelihood function and sum over all individuals
    LL = log(Prob_indiv)

    # Create the Biogeme object containing the object database and the formula for the contribution to the log-likelihood of each row using the following syntax:
    biogeme = bio.BIOGEME(database, LL)

    # The following syntax passes the name of the model:
    biogeme.modelName = model_name

    # Some object settings regaridng whether to save the results and outputs 
    biogeme.generate_pickle = False
    biogeme.generate_html   = False
    biogeme.save_iterations = False

    # Syntax to calculate the null log-likelihood. The null-log-likelihood is used to compute the rho-square 
    biogeme.calculate_null_loglikelihood(AV)

    # This line starts the estimation and returns the results object.
    results = biogeme.estimate()
    return results

def print_results(results):
    
    # Print the estimation statistics
    print(f'\n')
    print(results.short_summary())

    # Get the model parameters in a pandas table and  print it
    beta_hat = results.get_estimated_parameters()
    
    # Round the results to suitable decimal places
    beta_hat = beta_hat.round(4)
    beta_hat['Rob. t-test']  = beta_hat['Rob. t-test'].round(2)
    beta_hat['Rob. p-value'] = beta_hat['Rob. p-value'].round(2)
    print(beta_hat)

def create_collage(img_path,img_list,txt = None,cols = 5,rows = 4):
    '''
    Function to create a collage of images

    Parameters:
    img_path: path to the images
    img_list: list or pd.Series of image names
    txt: list or pd.Series of text to be displayed on the images
    cols: number of columns
    rows: number of rows
    '''

    # Convert to list if series
    if isinstance(img_list, pd.Series):
        img_list = img_list.tolist()

    # Convert to list if series
    if isinstance(txt, pd.Series):
        txt = txt.tolist()

    # GRID GENERATOR
    fig = plt.figure(figsize=(cols*12, rows*9))
    grid = ImageGrid(fig, 111, 
                    nrows_ncols=(rows, cols), 
                    axes_pad=0.1, 
                    )
    ii = 0
    for i in range(0,len(img_list)):
        
        if ii<(rows*cols):
            path = img_path / ''.join(img_list[i])
            try:
                img = Image.open(path)
                img = img.resize((200,133))
                grid[ii].imshow(img)
                
                # Add title
                if txt != None:
                    title_str = txt[ii]
                    # grid[ii].set_title(title_str,x =0.02,y=0.93,fontsize=30,loc = 'left')
                    title_obj = grid[ii].set_title(title_str, x=0.02, y=0.9, fontsize=30, loc='left')
                    title_obj.set_bbox(dict(facecolor='white', alpha=0.5))  # Adjust alpha for transparency

                grid[ii].set_xticks([])
                grid[ii].set_yticks([])

                ii = ii + 1
            except:
                print('File not found in. Image not loaded')
                print(path)
    plt.axis('off')
    plt.show()

def plot_distributions(results, distr_types, xmin, xmax):

    parts = list(distr_types)
    fig, ax = plt.subplots(
        1, len(parts),
        figsize=(6 * len(parts), 5),
        sharey=False
    )
    if len(parts) == 1:
        ax = [ax]

    x = np.linspace(xmin, xmax, 500)
    params = results.get_beta_values()
    
    for i, part in enumerate(parts):
        mu    = float(params[f'B_{part}'])
        sigma = abs(float(params[f'sigma_{part}']))

        spec   = distr_types[part]
        d_type = spec['dist']
        sign   = spec.get('sign', 1)              # default +1

        # ------------------------------------------------------------------
        # pick the pdf and mean formula for the requested distribution
        # ------------------------------------------------------------------
        if d_type == 'normal':
            mean = mu
            pdf  = (1 / (sigma * np.sqrt(2*np.pi))
                   ) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        elif d_type == 'lognormal':
            raw_mean = np.exp(mu + 0.5 * sigma ** 2)   # > 0 by definition
            mean     = sign * raw_mean

            # Base pdf lives on the positive real line
            x_pos    = np.abs(x)
            base_pdf = (1 / (x_pos * sigma * np.sqrt(2*np.pi))
                       ) * np.exp(-((np.log(x_pos) - mu) ** 2) / (2 * sigma ** 2))

            # Flip to the requested side of the axis
            if sign < 0:
                pdf = np.where(x < 0, base_pdf, 0)
            else:
                pdf = np.where(x > 0, base_pdf, 0)

        else:
            raise ValueError(f"Unknown distribution type: {d_type}")

        # ------------------------------------------------------------------
        # draw
        # ------------------------------------------------------------------
        ax[i].plot(x, pdf, label=d_type)
        ax[i].axvline(0, ls='--', c='k')
        ax[i].axhline(0, ls='-', c='k')

        ax[i].set_title(f'{part}: {d_type}\nmean = {mean:.3g}')
        ax[i].set_xlabel('Marginal utility')
        if i == 0:
            ax[i].set_ylabel('PDF')
        ax[i].legend()

    plt.tight_layout()
    plt.show()
