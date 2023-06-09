# README

ClimSIPS is a selection protocol designed to select subsets of CMIP5 or CMIP6 models for downstream applications. Results are presented in a ternary contour plot to illustrate how ascribing different levels of priority to performance, independence, and spread affects subset composition. 

The package imports performance, independence, and spread predictors. For the European case studies example, predictors are available here: 
https://www.research-collection.ethz.ch/handle/20.500.11850/599312.


Metrics are computed from the predictors and formatted for use in the selection step. Metrics are plotted for user edification:
	- model performance, in order from highest performing (closest to observations) to lowest performing
	- model independence, displaying the distance of each model to all others in the ensemble
	- spread, as a scatter of the targeted projection variables (midcentury regional/seasonal average temperature vs. precipitation change)

The selection step computes a cost function for each set of models (m choose n combinations). The cost function is comprised of three terms (performance, independence, and spread) and two parameters (alpha and beta) that set the relative importance of each term. The set of models that minimizes the cost function for each combination of alpha and beta is returned. 

The current code allows the user to specify the following:

- selection from 'CMIP5' or 'CMIP6' (models listed in Merrifield et al. 2022)
- flexibility of custom CMIP5 and CMIP6 starting ensembles
- models represented by their ensemble mean (EM) or by an individual member (IM; selected to maximizes overall spread in the ensemble)
- region and season of targeted projection and performance predictors; currently JJA_CEU, DJF_CEU, and DJF_NEU implemented
- size of desired subset (m)
- resolution of the ternary contour plot (alpha and beta) 
- a performance threshold to filter out lower performing models prior to the selection step (perf_cutoff)
- an option to run the selection step in parallel on multiple cores (max_workers)
- option to output the minimum or the next to minimum of the cost function (min2)

To run ClimSIPS, please set the following inputs in main.py:
   
    ############# INPUTS for pre-processing #############
    # paths to predictors
    perf_path= '/***/predictors/performance/
    spread_path='/***/predictors/spread/'
    indep_path='/***/predictors/independence/'
    # ensemble and representation
    cmip = 'CMIP6' 
    im_or_em = 'IM'
    season_region = 'JJA_CEU' 
    #####################################################


    ############# INPUTS for subselection #############
    m = 2 # number of models in the subset
    alpha = 10 # number of steps in alpha's [0,1] range
    beta = 10 # number of steps in alpha's [0,1] range
    perf_cutoff = 2 # performance threshold to pre-filter models (if desired)
    max_workers = 1
    min2 = False
    ###################################################

