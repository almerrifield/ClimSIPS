
### README

ClimSIPS is a selection protocol designed to select subsets of CMIP5 or CMIP6 models for downstream applications. Results are presented in a ternary contour plot to illustrate how ascribing different levels of priority to performance, independence, and spread affects subset composition. 

### Installation
The current paper release of ClimSIPS runs in the environment specified in ClimSIPS.yml. To access the enviroment, please use the following command:

`conda env create -f ClimSIPS.yml`

`source activate ClimSIPS`

* Note: This environment is pinned for the paper.

We have also provided two alternative development environments.

requirements.txt can be pip installed as:

`pip install -r requirements.txt`

or ClimSIPS_dev can be accessed with:

`conda env create -f ClimSIPS_dev.yml`

`source activate ClimSIPS_dev`

### Data
The package imports performance, independence, and spread predictors. For the European case studies example, predictors are available here: 
https://www.research-collection.ethz.ch/handle/20.500.11850/599312.
* predictors for new cases (e.g., DJF_CEU, JJA_CH, and DJF_CH) will be made available upon request.

### Usage
Metrics are computed from the predictors and formatted for use in the selection step. Metrics are plotted for user edification:
- model performance, in order from highest performing (closest to observations) to lowest performing
- model independence, displaying the distance of each model to all others in the ensemble
- spread, as a scatter of the targeted projection variables (midcentury regional/seasonal average temperature vs. precipitation change)

The selection step computes a cost function for each set of models (m choose n combinations). The cost function is comprised of three terms (performance, independence, and spread) and two parameters (alpha and beta) that set the relative importance of each term. The set of models that minimizes the cost function for each combination of alpha and beta is returned. 

The current code allows the user to specify the following:

- selection from 'CMIP5' or 'CMIP6' (models listed in Merrifield et al. 2022)
- flexibility of custom CMIP5 and CMIP6 starting ensembles in member_selection.py
- models represented by their ensemble mean (EM) or by an individual member (IM; selected to maximize overall spread in the ensemble)
- region and season of targeted projection and performance predictors; currently JJA_CEU, DJF_CEU, and DJF_NEU implemented
- size of desired subset (m)
- resolution (step size) of the ternary contour plot (alpha and beta) 
- a performance threshold to filter out lower performing models prior to the selection step (perf_cutoff)
- an option to run the selection step in parallel on multiple cores (max_workers)
- option to output the minimum or the next to minimum of the cost function (min2)

To run ClimSIPS, please pass the path to your predictors directory, e.g.:

`python main.py /home/data/predictors/`

and set the following inputs in main.py:

``` python
    ############# INPUTS for pre-processing #############
    # ensemble and representation
    cmip = 'CMIP5' # 'CMIP5' or 'CMIP6'
    im_or_em = 'IM' # 'IM' individual member or 'EM' ensemble mean
    season_region = 'JJA_CEU' # 'JJA_CEU' or 'DJF_NEU'
    #####################################################

    ############# INPUTS for subselection #############
    m = 2 # number of models in the subset
    alpha = 10 # number of steps in alpha's [0,1] range
    beta = 10 # number of steps in alpha's [0,1] range
    perf_cutoff = 2 # performance threshold to pre-filter models (if desired)
    max_workers = 1
    min2 = False
    ###################################################
```
