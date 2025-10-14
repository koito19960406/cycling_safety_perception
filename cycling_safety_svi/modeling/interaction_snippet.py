
def create_demographic_dummy_variables(self, data):
    """Create dummy variables for demographic categories and safety interactions."""
    print(f"Creating dummy variables and interaction terms for group: {self.model_group}...")
    
    interaction_features = []
    
    def create_dummies(feature, ref_cat):
        cat_col = f'{feature}_cat'
        if cat_col not in data.columns: return []
        
        cats = [c for c in data[cat_col].unique() if pd.notna(c) and c != ref_cat]
        for cat in cats:
            # Labels from demographic_mappings are already sanitized
            dummy_name = cat
            
            data[f'{dummy_name}_1'] = (data[cat_col] == cat).astype(int)
            data[f'{dummy_name}_2'] = data[f'{dummy_name}_1']
            
            interact_name = f"safety_{dummy_name}"
            data[f'{interact_name}_1'] = data['SAFETY_SCORE1'] * data[f'{dummy_name}_1']
            data[f'{interact_name}_2'] = data['SAFETY_SCORE2'] * data[f'{dummy_name}_2']
            interaction_features.append(interact_name)
        return cats

    for demo in self.demographic_variables:
        # The logic to determine the reference category is now centralized here.
        # It finds the lowest numeric key present in the actual data for that demographic.
        if f'{demo}_cat' in data.columns and data[f'{demo}_cat'].notna().any():
            
            # Get the numeric keys corresponding to the categories present in the data
            reverse_mapping = {v: k for k, v in self.demographic_mappings[demo].items()}
            present_cats = data[f'{demo}_cat'].dropna().unique()
            data_keys = [reverse_mapping[cat] for cat in present_cats if cat in reverse_mapping]
            
            if data_keys:
                ref_key = min(data_keys)
                ref_cat = self.demographic_mappings[demo][ref_key]
                create_dummies(demo, ref_cat)
            else:
                print(f"Warning: Could not find a valid reference category for '{demo}'. Skipping dummy creation.")

    return data, interaction_features

def estimate_interaction_model(self):
    """Estimate the MXL safety * demographics interaction model."""
    print("\nEstimating Safety * Demographics Interaction Model (MXL)...")
    
    model_data = self.merged_data.copy()
    
    model_data, interaction_features = self.create_demographic_dummy_variables(model_data)

    features = self.original_model_features + interaction_features + ['SAFETY_SCORE']
    
    # Prepare panel data
    _, biodata_wide, obs_per_ind = prepare_panel_data(model_data, self.individual_id, 'CHOICE', features)
    
    # Define parameters and utility
    random_params_config = {
        'TT': {'mean_init': -1, 'sigma_init': 0.1, 'dist': 'lognormal'}, 
        'TL': {'mean_init': -1, 'sigma_init': 0.1, 'dist': 'lognormal'},
        'SAFETY_SCORE': {'mean_init': 1.0, 'sigma_init': 0.1, 'dist': 'normal'}
    }
    random_params = {}
    for p, c in random_params_config.items():
        mean = Beta(f'B_{p}', c['mean_init'], None,None,0)
        sigma = Beta(f'sigma_{p}', c['sigma_init'], None,None,0)
        draws = bioDraws(f'{p}_rnd', 'NORMAL_HALTON2')
        if c.get('dist') == 'lognormal':
            random_params[p] = -exp(mean + sigma * draws)
        else:
            random_params[p] = mean + sigma * draws

    fixed_features = self.original_model_features + interaction_features
    fixed_params = {f: Beta(f"B_{f.replace(' - ', '___').replace(' ', '_')}", 0, None,None,0) for f in fixed_features}

    # Create utility function
    V = []
    for q in range(obs_per_ind):
        V1, V2 = 0, 0
        # Random parameters
        for name, param in random_params.items():
            scale = 10 if name == 'TT' else (3 if name == 'TL' else 1)
            v1_name, v2_name = f"{name}1_{q}", f"{name}2_{q}"
            if v1_name in biodata_wide.variables:
                V1 += param * Variable(v1_name) / scale
                V2 += param * Variable(v2_name) / scale
        # Fixed parameters
        for name, param in fixed_params.items():
            v1_name, v2_name = f"{name}_1_{q}", f"{name}_2_{q}"
            if v1_name in biodata_wide.variables:
                V1 += param * Variable(v1_name)
                V2 += param * Variable(v2_name)
            else:
                print(f"Warning: Variable {v1_name} not found in biodata_wide, skipping.")
        V.append({1: V1, 2: V2})

    # Estimate model
    model_name = f'demographics_interaction_{self.model_group}'
    results = estimate_mxl(V, {1:1, 2:1}, 'CHOICE', obs_per_ind, self.num_draws, biodata_wide, model_name, self.output_dir)
    
    self.results = (results, obs_per_ind)
    print_mxl_results(results)


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
