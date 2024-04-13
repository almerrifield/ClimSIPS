# main.py

#################################
# packages
#################################

import ClimSIPS.function as csf
import ClimSIPS.pre_processing as cspp
import ClimSIPS.plots as csp

import sys
import configparser

def main():
    print('==== starting subselection ====')

    ############# INPUTS for pre-processing #############

    all_config = configparser.ConfigParser()

    if len(sys.argv) == 4:
        all_config.read(sys.argv[1])
        config_key = sys.argv[2]
        predictors_root = sys.argv[3]
    if len(sys.argv) == 3:
        all_config.read(sys.argv[1])
        config_key = sys.argv[2]
        predictors_root = "/net/h2o/climphys/meranna/Data/predictors/"
    else:
        print("Use python main.py config_climsips.ini config_section_key predictor_root_directory")

    config = all_config[config_key]
    skip_preprocessing_with=config.get('skip_preprocessing_with', fallback='')

    # ensemble and representation
    cmip = config['cmip']
    im_or_em = config.get('im_or_em','IM')
    season_region = config['season_region']
    double_norm = config.getboolean('double_norm',fallback=False)

    # convert subselection inputs in the config to integers
    m = config.getint('m')
    alpha = config.getint('alpha_steps',fallback=10)
    beta = config.getint('beta_steps',fallback=10)
    perf_cutoff = config.getint('perf_cutoff',fallback=10)
    max_workers = config.getint('max_workers',fallback=1)
    min2 = config.getboolean('min2')

#####################################################

    if skip_preprocessing_with == '':
        # paths to predictors
        perf_path=predictors_root+'performance/'
        spread_path=predictors_root+'spread/'
        indep_path=predictors_root+'independence/'

        #  pre-processing: obtain performance, independence, and spread metrics
        # TO DO: implement scenarios
        scenario='SSP585'
        common_models = cspp.model_soup(perf_path,indep_path,spread_path, cmip, im_or_em, season_region,double_norm,scenario)
        #print(common_models)
        dsDeltaQ = cspp.pre_process_perf(perf_path, cmip, im_or_em, season_region,spread_path,double_norm=double_norm,default_models=common_models)
        ds_spread_metric,targets = cspp.pre_process_spread(spread_path, cmip, im_or_em, season_region,default_models=common_models)
        dsWi = cspp.pre_process_indep(indep_path, cmip, im_or_em, season_region,spread_path,default_models=common_models)

        # save output file
        outfile = 'perf_ind_spread_metrics.nc'
        csf.make_output_file(dsDeltaQ,ds_spread_metric,targets,dsWi,outfile)
    else:
        outfile = skip_preprocessing_with
        print('skipping preprocessing with', outfile)

    # plot components
    csp.performance_order(outfile,cmip,im_or_em,season_region,plotname="performance_order.png")
    csp.independence_square(outfile,cmip,im_or_em,season_region,plotname="independence_metric.png")
    csp.spread_scatter(outfile,cmip,im_or_em,season_region,plotname="spread_scatter.png")

    # subselection
    optimal_models_csv = csf.select_models(outfile, cmip, im_or_em, season_region, m, alpha, beta, perf_cutoff, max_workers=max_workers, min2=min2)

    csp.selection_triangle(optimal_models_csv,alpha,plotname="optimal_subsets.png")

    print('---- subselection complete ----')


if __name__ == '__main__':
    main()
