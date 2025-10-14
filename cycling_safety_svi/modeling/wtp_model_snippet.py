    def _estimate_wtp_mxl(self, wtp_attribute, cost_attribute, model_name):
        """
        Estimate a separate mixed logit model in WTP space.
        Following the reference Lab_02A approach where WTP models are completely separate
        from utility space models, with their own parameter definitions.
        
        Includes all significant segmentation features from backward elimination.
        
        Args:
            wtp_attribute: The attribute we want to compute WTP for (e.g., 'SAFETY_SCORE')
            cost_attribute: The cost attribute (e.g., 'TT' or 'TL')  
            model_name: Name of the model
        """
        from mxl_functions import estimate_wtp_mxl, prepare_panel_data
        from biogeme.expressions import Beta, Variable, bioDraws, exp
        
        # Prepare data - include cost, WTP, and segmentation attributes
        attributes = [self.individual_id, 'CHOICE']
        
        # Add cost attribute columns
        if cost_attribute == 'TT':
            attributes.extend(['TT1', 'TT2'])
            cost_scale = 10
        else:  # TL
            attributes.extend(['TL1', 'TL2'])
            cost_scale = 3
            
        # Add WTP attribute columns    
        if wtp_attribute == 'SAFETY_SCORE':
            attributes.extend(['safety_score_1', 'safety_score_2'])
        
        # Add significant segmentation features
        if hasattr(self, 'final_significant_features'):
            seg_features = [f for f in self.final_significant_features if f not in ['SAFETY_SCORE', 'TT', 'TL']]
            for feature in seg_features:
                attributes.extend([f"{feature}_1", f"{feature}_2"])
            print(f"Including {len(seg_features)} segmentation features in WTP model: {seg_features}")
        else:
            seg_features = []
            print("Warning: No final_significant_features found, WTP model will only include safety and cost")
            
        model_data = self.merged_data[attributes].copy().dropna()
        
        # Rename safety score columns to match expected format
        if wtp_attribute == 'SAFETY_SCORE':
            model_data = model_data.rename(columns={
                'safety_score_1': 'SAFETY_SCORE1',
                'safety_score_2': 'SAFETY_SCORE2'
            })

        _, biodata_wide, obs_per_ind = prepare_panel_data(
            model_data, self.individual_id, 'CHOICE'
        )

        # Define WTP space parameters following Lab_02A reference implementation
        # Parameters for log-normal WTP distribution
        mu = Beta('mu', -1, None, None, 0)  # Starting value from reference
        sigma = Beta('sigma', 1, None, None, 0)  # Starting value from reference
        
        # Cost parameter (fixed, negative) - this is like B_tc in the reference
        B_cost = Beta('B_cost', -0.1, None, None, 0)  # Starting value from reference
            
        # Random WTP parameter (log-normal distribution)
        # This is like vtt_rnd in the reference: vtt_rnd = exp(mu + sigma * bioDraws('vtt_rnd', 'NORMAL_HALTON2'))
        wtp_rnd = -exp(mu + sigma * bioDraws('wtp_rnd', 'NORMAL_HALTON2'))

        # Define fixed parameters for segmentation features
        fixed_params = {}
        for feature in seg_features:
            beta_name = self._sanitize_name_for_beta(feature)
            fixed_params[feature] = Beta(beta_name, 0, None, None, 0)

        # Define utility functions in WTP space following Lab_02A approach:
        # V_L = B_tc * (CostL + vtt_rnd * TimeL) + segmentation_terms
        # V_R = B_tc * (CostR + vtt_rnd * TimeR) + segmentation_terms
        V = []
        for q in range(obs_per_ind):
            # Get variable names for this observation
            cost1_name = f"{cost_attribute}1_{q}"
            cost2_name = f"{cost_attribute}2_{q}"
            wtp1_name = f"{wtp_attribute}1_{q}"
            wtp2_name = f"{wtp_attribute}2_{q}"
            
            V1 = V2 = 0
            
            # Build WTP space utility: B_cost * (Cost + WTP * WTP_attribute)
            if (cost1_name in biodata_wide.variables and 
                wtp1_name in biodata_wide.variables):
                
                # Following Lab_02A: V = B_cost * (Cost + WTP_rnd * WTP_attribute)
                V1 = B_cost * (Variable(cost1_name) / cost_scale + wtp_rnd * Variable(wtp1_name))
                V2 = B_cost * (Variable(cost2_name) / cost_scale + wtp_rnd * Variable(wtp2_name))
                
                # Add segmentation features (outside the WTP factorization)
                for feature in seg_features:
                    var1_name = f"{feature}_1_{q}"
                    var2_name = f"{feature}_2_{q}"
                    if var1_name in biodata_wide.variables and var2_name in biodata_wide.variables:
                        V1 += fixed_params[feature] * Variable(var1_name)
                        V2 += fixed_params[feature] * Variable(var2_name)
            
            V.append({1: V1, 2: V2})

        # Estimate the model using the WTP-specific estimation function
        results = estimate_wtp_mxl(V, {1: 1, 2: 1}, 'CHOICE', obs_per_ind, 
                                  self.num_draws, biodata_wide, model_name, self.output_dir)
        
        return results.data, obs_per_ind, cost_attribute, wtp_attribute


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