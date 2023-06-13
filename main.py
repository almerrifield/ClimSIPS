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
    pred_path = "./data/predictors/"
    perf_path = pred_path + 'performance/'
    spread_path = pred_path + 'spread/'
    indep_path =pred_path + 'independence/'
    # ensemble and representation
    cmip = 'CH202x'
    im_or_em = 'IM'
    season_region = 'DJF_CEU'
    #####################################################

    #  pre-processing: obtain performance, independence, and spread metrics
    dsDeltaQ = cspp.pre_process_perf(perf_path, cmip, im_or_em, season_region)
    ds_spread_metric,targets = cspp.pre_process_spread(spread_path, cmip, im_or_em, season_region)
    dsWi = cspp.pre_process_indep(indep_path, cmip, im_or_em, season_region)

    # save output file
    outfile = 'perf_ind_spread_metrics.nc'
    csf.make_output_file(dsDeltaQ,ds_spread_metric,targets,dsWi,outfile)

    # plot components
    csp.performance_order(outfile,cmip,im_or_em,season_region,plotname="performance_order.png")
    csp.independence_square(outfile,cmip,im_or_em,season_region,plotname="independence_metric.png")
    csp.spread_scatter(outfile,cmip,im_or_em,season_region,plotname="spread_scatter.png")


    ############# INPUTS for subselection #############
    m = 5 # number of models in the subset
    alpha = 10 # number of steps in alpha's [0,1] range
    beta = 10 # number of steps in alpha's [0,1] range
    perf_cutoff = 2 # performance threshold to pre-filter models (if desired)
    max_workers = 1
    min2 = False
    ###################################################

    optimal_models_csv = csf.select_models(outfile, cmip, im_or_em, season_region, m, alpha, beta, perf_cutoff, max_workers=max_workers, min2=min2)

    csp.selection_triangle(optimal_models_csv,alpha,plotname="optimal_subsets.png")

    print('---- subselection complete ----')


if __name__ == '__main__':
    main()
