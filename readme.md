
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

To run ClimSIPS with the pre-processor, the package imports performance, independence, and spread predictors. For the European case studies example, predictors are available here:
https://www.research-collection.ethz.ch/handle/20.500.11850/599312.
* predictors for new cases (e.g., DJF_CEU, JJA_CH, and DJF_CH) will be made available upon request.

### Usage

To run ClimSIPS, please set the following inputs in config_climsips.ini:

``` python
[default]
#### pre-processing inputs ####

skip_preprocessing_with=precomputed_predictor_outfiles/perf_ind_spread_metrics_CMIP6_EM_JJA_CEU.nc

# ensemble options: CMIP5, CMIP6
cmip = CMIP6

# member options: EM, IM
im_or_em = EM

# region/season options: JJA_CEU, DJF_NEU
season_region = JJA_CEU

# normalize performance with other CMIP ensemble
double_norm =yes

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

To run the package without the preprocessor:

```python
python main.py config_climsips.ini default
```

To run the package with the preprocessor:
```python
  #### pre-processing inputs ####
  
  skip_preprocessing_with=
  ...
```

```python
python main.py config_climsips.ini default path_to_predictors
```

Metrics are computed from the predictors and formatted for use in the selection step. Metrics are plotted for user edification:
- model performance, in order from highest performing (closest to observations) to lowest performing
- model independence, displaying the distance of each model to all others in the ensemble
- spread, as a scatter of the targeted projection variables (midcentury regional/seasonal average temperature vs. precipitation change)

The selection step computes a cost function for each set of models (m choose n combinations). The cost function is comprised of three terms (performance, independence, and spread) and two parameters (alpha and beta) that set the relative importance of each term. The set of models that minimizes the cost function for each combination of alpha and beta is returned.

The current code allows the user to specify the following:

- selection from 'CMIP5' or 'CMIP6' (models listed in Merrifield et al. 2022)
- models represented by their ensemble mean (EM) or by an individual member (IM; selected to maximize overall spread in the ensemble)
- region and season of targeted projection and performance predictors; currently JJA_CEU and DJF_NEU available
- size of desired subset (m)
- resolution (step size) of the ternary contour plot (alpha and beta)
- a performance threshold to filter out lower performing models prior to the selection step (perf_cutoff)
- an option to run the selection step in parallel on multiple cores (max_workers)
- option to output the minimum or the next to minimum of the cost function (min2)
