# main.py

#################################
# packages
#################################

import ClimSIPS.function as csf
import ClimSIPS.pre_processing as cspp
import ClimSIPS.plots as csp


def main():
    print('==== starting subselection ====')

    ############# INPUTS for pre-processing #############
    # paths to predictors
    perf_path='/net/h2o/climphys/meranna/Data/predictors/performance/'
    spread_path='/net/h2o/climphys/meranna/Data/predictors/spread/'
    indep_path='/net/h2o/climphys/meranna/Data/predictors/independence/'
    # ensemble and representation
    cmip = 'CMIP6'
    im_or_em = 'IM'
    #####################################################

    #  pre-processing: obtain performance, independence, and spread metrics
    dsDeltaQ = cspp.pre_process_perf(perf_path, cmip, im_or_em)
    ds_spread_metric,targets = cspp.pre_process_spread(spread_path, cmip, im_or_em)
    dsWi = cspp.pre_process_indep(indep_path, cmip, im_or_em)

    # save output file
    outfile = 'perf_ind_spread_metrics.nc'
    csf.make_output_file(dsDeltaQ,ds_spread_metric,targets,dsWi,outfile)

    # plot components
    csp.performance_order(outfile,cmip,im_or_em,plotname="performance_order.png")
    csp.independence_square(outfile,cmip,im_or_em,plotname="independence_metric.png")
    csp.spread_scatter(outfile,cmip,im_or_em,plotname="spread_scatter.png")


    ############# INPUTS for subselection #############
    m = 2 # number of models in the subset
    alpha = 10 # number of steps in alpha's [0,1] range
    beta = 10 # number of steps in alpha's [0,1] range
    perf_cutoff = 2 # performance threshold to pre-filter models (if desired)
    max_workers = 1
    ###################################################

    optimal_models_csv = csf.select_models(outfile, cmip, im_or_em, m, alpha, beta, perf_cutoff, max_workers=max_workers)

    csp.selection_triangle(optimal_models_csv,plotname="optimal_subsets.png")

    print('---- subselection complete ----')


if __name__ == '__main__':
    main()
