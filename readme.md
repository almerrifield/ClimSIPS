
### README

ClimSIPS is a selection protocol designed to select subsets of CMIP5 or CMIP6 models for downstream applications. Results are presented in a ternary contour plot to illustrate how ascribing different levels of priority to performance, independence, and spread affects subset composition.

### Installation

We recommend running ClimSIPS in the development environment:  

ClimSIPS_dev can be accessed with:

`conda env create -f ClimSIPS_dev.yml`

`source activate ClimSIPS_dev`

Alternatively, requirements.txt can be pip installed as:

`pip install -r requirements.txt`



The paper release of ClimSIPS runs in the environment specified in ClimSIPS.yml. The enviroment may be difficult to resolve for users external to ETH Zurich.


### Data
ClimSIPS can be run with or without the preprocessor.

To run ClimSIPS without the preprocesso

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

To run ClimSIPS, please set the following inputs in config_climsips.ini:

``` python
    #### pre-processing inputs ####
    # ensemble options: CMIP5, CMIP6, CH202x, RCM
    cmip = CMIP6

    # member options: EM, IM
    im_or_em = IM

    # region/season options: JJA_CEU, JJA_CH, DJF_NEU, DJF_CEU, DJF_CH
    season_region = JJA_CEU

    #### subselection inputs ####
    # number of models in the subset
    m = 2

    # number of steps in alpha's [0,1] range
    alpha_steps = 5
    beta_steps = 5

    # performance threshold to pre-filter models (if desired)
    perf_cutoff = 10

    # setting for parallel processing
    max_workers = 1

    # find the secondary minimum of the cost function
    min2 = False
```

The package runs with the following:

```python
python main.py config_climsips.ini default path_to_predictors
```
