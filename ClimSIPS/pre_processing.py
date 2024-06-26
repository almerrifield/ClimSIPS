import xarray as xr
import numpy as np
import functools
from functools import reduce
from . import function as csf

# ################################
# Selecting Common Model Set
# ################################

def model_soup(perf_path,indep_path,spread_path, cmip, im_or_em, season_region,double_norm,scenario):
    perf = pre_process_perf_load_delta(perf_path, cmip, season_region,default_models=False)
    spread = pre_process_spread_load(spread_path, cmip, season_region, default_models=False)
    indep = pre_process_indep_load(indep_path, cmip, default_models=False)

    models = []
    for x in perf:
        models.append(list(x.member.data))
    for x in spread:
        models.append(list(x.member.data))
    for x in indep:
        models.append(list(x.member.data))

    common = reduce(np.intersect1d,models)
    print('Initial Ensemble Size:', len(common))
    return tuple(common)


def metric_information(cmip, im_or_em, season_region):
    if cmip == 'CMIP5' and season_region == 'JJA_CEU':
        print("Performance Fields: tos, swcre, tas, pr")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CMIP5' and season_region == 'JJA_CH':
        print("Performance Fields: tos, swcre, tas, pr")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CMIP5' and season_region == 'DJF_NEU':
        print("Performance Fields: tos, swcre, tas, pr, psl")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CMIP5' and season_region == 'DJF_CEU':
        print("Performance Fields: tos, swcre, tas, pr, psl")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CMIP5' and season_region == 'DJF_CH':
        print("Performance Fields: tos, swcre, tas, pr, psl")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CH202x' and season_region == 'JJA_CEU':
        print("Performance Fields: tas")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CH202x' and season_region == 'JJA_CH':
        print("Performance Fields: tos, tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CH202x' and season_region == 'DJF_NEU':
        print("Performance Fields: tos, tas, pr, psl")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CH202x' and season_region == 'DJF_CEU':
        print("Performance Fields: tos, tas, pr, rsds")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CH202x' and season_region == 'DJF_CH':
        print("Performance Fields: tos, tas, pr, psl")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM' and season_region == 'JJA_ALPS':
        print("Performance Fields: tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM' and season_region == 'JJA_CH':
        print("Performance Fields: tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM' and season_region == 'JJA_CEU':
        print("Performance Fields: tas")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM' and season_region == 'DJF_ALPS':
        print("Performance Fields: tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM' and season_region == 'DJF_CH':
        print("Performance Fields: tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM' and season_region == 'DJF_CEU':
        print("Performance Fields: tas")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CMIP6' and season_region == 'JJA_CEU':
        print("Performance Fields: tos, swcre, tas, pr")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CMIP6' and season_region == 'JJA_CH':
        print("Performance Fields: tos, swcre, tas, pr")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CMIP6' and season_region == 'DJF_NEU':
        print("Performance Fields: tos, swcre, tas, pr, psl")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CMIP6' and season_region == 'DJF_CEU':
        print("Performance Fields: tos, swcre, tas, pr, psl")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CMIP6' and season_region == 'DJF_CH':
        print("Performance Fields: tos, swcre, tas, pr, psl")
        print("Spread change period: 2041-2060 - 1995-2014; SSP585")
    if cmip == 'CH202x_CMIP6' and season_region == 'JJA_CEU':
        print("Performance Fields: tas")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CH202x_CMIP6' and season_region == 'JJA_CH':
        print("Performance Fields: tos, tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_NEU':
        print("Performance Fields: tos, tas, pr, psl")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_CEU':
        print("Performance Fields: tos, tas, pr, rsds")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_CH':
        print("Performance Fields: tos, tas, pr, psl")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM_CMIP6' and season_region == 'JJA_ALPS':
        print("Performance Fields: tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM_CMIP6' and season_region == 'JJA_CH':
        print("Performance Fields: tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM_CMIP6' and season_region == 'JJA_CEU':
        print("Performance Fields: tas")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM_CMIP6' and season_region == 'DJF_ALPS':
        print("Performance Fields: tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM_CMIP6' and season_region == 'DJF_CH':
        print("Performance Fields: tas, pr")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")
    if cmip == 'RCM_CMIP6' and season_region == 'DJF_CEU':
        print("Performance Fields: tas")
        print("Spread change period: 2070-2099 - 1981-2010; SSP585")

# ################################
# Performance
# ################################

def pre_process_perf(path, cmip, im_or_em, season_region, spread_path, default_models, double_norm=False):
    deltas = pre_process_perf_load_delta(path, cmip, season_region, default_models=default_models)

    if double_norm:
        normalization_partner = {
            'CMIP5': 'CMIP6',
            'CMIP6': 'CMIP5',
            'CH202x': 'CH202x_CMIP6',
            'CH202x_CMIP6': 'CH202x',
            'RCM': 'RCM_CMIP6',
            'RCM_CMIP6': 'RCM',
        }
        other = normalization_partner[cmip]
        print('normalized with', other)
        deltas_other = pre_process_perf_load_delta(path, other, season_region, default_models=False)
    else:
        print('with self normalization')
        deltas_other = deltas
    return pre_process_perf_rest(deltas, deltas_other, cmip, im_or_em, season_region,spread_path,default_models=default_models)

@functools.lru_cache
def pre_process_perf_load_delta(path, cmip, season_region,default_models):
    if cmip not in ['CMIP5','CMIP6','CH202x','CH202x_CMIP6','RCM','RCM_CMIP6']:
        raise NotImplementedError(cmip)

    ##################################################
    # CMIP5 Predictors
    ##################################################

    ## CMIP5 JJA CEU
    if cmip == 'CMIP5' and season_region == 'JJA_CEU':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP5_rcp85_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_EUR_ann_1950-1969_mean.nc'
        SW_jja_fn = 'swcre_mon_CMIP5_rcp85_g025_CEU_jja_2001-2018_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSW_jja_base = csf.load_models(path,SW_jja_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        SWobs_jja_fn = 'swcre_mon_OBS_g025_CEU_jja_2001-2018_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSWobs_jja_base = csf.load_observations(path,SWobs_jja_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSW_jja_base_delta = csf.compute_predictor_deltas(dsSW_jja_base,dsSWobs_jja_base,'swcre')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSW_jja_base_delta, dsPr_jja_base_delta]

    ## CMIP5 JJA CH
    if cmip == 'CMIP5' and season_region == 'JJA_CH':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP5_rcp85_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_CH_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_CH_ann_1950-1969_mean.nc'
        SW_jja_fn = 'swcre_mon_CMIP5_rcp85_g025_CH_jja_2001-2018_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_CH_jja_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSW_jja_base = csf.load_models(path,SW_jja_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_CH_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_CH_ann_1950-1969_mean.nc'
        SWobs_jja_fn = 'swcre_mon_OBS_g025_CH_jja_2001-2018_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_CH_jja_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSWobs_jja_base = csf.load_observations(path,SWobs_jja_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSW_jja_base_delta = csf.compute_predictor_deltas(dsSW_jja_base,dsSWobs_jja_base,'swcre')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSW_jja_base_delta, dsPr_jja_base_delta]

    ## CMIP5 DJF NEU
    if cmip == 'CMIP5' and season_region == 'DJF_NEU':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP5_rcp85_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_EUR_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP5_rcp85_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ## CMIP5 DJF CEU
    if cmip == 'CMIP5' and season_region == 'DJF_CEU':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP5_rcp85_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_EUR_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP5_rcp85_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-CEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-CEU_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ## CMIP5 DJF CH
    if cmip == 'CMIP5' and season_region == 'DJF_CH':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP5_rcp85_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_CH_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_CH_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP5_rcp85_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_CH_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_CH_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_CH_ann_1950-1969_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_CH_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]


    ##################################################
    # CH202x (CMIP5) Predictors
    ##################################################

    ## CH202x JJA CEU
    if cmip == 'CH202x' and season_region == 'JJA_CEU':
        T_fn = 'tas_mon_CMIP5_rcp85_g025_CEU_jja_1971-2015_trend.nc'

        # load, filter for common models
        dsT_trnd = csf.load_models(path,T_fn,cmip,default_models=default_models)

        # load observations
        Tobs_fn = 'tas_mon_OBS_g025_CEU_jja_1971-2015_trend.nc'
        dsTobs_trnd = csf.load_observations(path,Tobs_fn)

        # compute predictor-obs RMSEs
        dsT_trnd_delta = csf.compute_predictor_deltas(dsT_trnd,dsTobs_trnd,'tas')

        return [dsT_trnd_delta]

    ## CH202x JJA CH
    if cmip == 'CH202x' and season_region == 'JJA_CH':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1981-2010_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_hist_g025_CH_ann_2001-2010_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_CH_jja_1981-2010_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1981-2010_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_CH_ann_2001-2010_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_OBS_g025_CH_jja_1981-2010_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsPr_jja_base_delta]

    ## CH202x DJF NEU
    if cmip == 'CH202x' and season_region == 'DJF_NEU':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_EUR_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP5_rcp85_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ## CH202x DJF CEU  (patch: test case)
    if cmip == 'CH202x' and season_region == 'DJF_CEU':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_EUR_ann_1950-1969_mean.nc'
        R_ann_fn = 'rsds_mon_CMIP5_rcp85_g025_CEU_ann_2001-2018_mean.nc'
        Pr_djf_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-CEU_djf_1995-2014_mean.nc'
        Pr_jja_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsR_ann_base = csf.load_models(path,R_ann_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_djf_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_jja_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        Robs_ann_fn = 'rsds_mon_OBS_g025_CEU_ann_2001-2018_mean_flip.nc'
        Probs_djf_fn = 'pr_mon_OBS_g025_EOBS-CEU_djf_1995-2014_mean.nc'
        Probs_jja_fn = 'pr_mon_OBS_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsRobs_ann_base = csf.load_observations(path,Robs_ann_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_djf_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_jja_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsR_ann_base_delta = csf.compute_predictor_deltas(dsR_ann_base,dsRobs_ann_base,'rsds')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsR_ann_base_delta, dsPr_djf_base_delta,dsPr_jja_base_delta]

    ## CH202x DJF CH
    if cmip == 'CH202x' and season_region == 'DJF_CH':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1981-2010_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_hist_g025_CH_ann_2001-2010_1971-1980_diff.nc'
        SLP_djf_fn = 'psl_mon_CMIP5_rcp85_g025_NATL_djf_1971-2010_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_CH_djf_1981-2010_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1981-2010_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_CH_ann_2001-2010_1971-1980_diff.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1971-2010_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_CH_djf_1981-2010_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]


    ##################################################
    # RCM (CMIP5) Predictors
    ##################################################

    ## RCM JJA ALPS (patch: currently test case)
    if cmip == 'RCM' and season_region == 'JJA_ALPS':
        T_base_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_alps_jja_1971-2020_mean.nc'
        T_diff_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_alps_ann_2011-2020_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_alps_ann_2011-2020_1971-1980_diff.nc'

        # load, filter for common models
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_diff = csf.load_models(path,T_diff_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        Tobs_base_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_jja_1971-2020_mean.nc'
        Tobs_diff_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_ann_2011-2020_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_ann_2011-2020_1971-1980_diff.nc'

        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_diff = csf.load_observations(path,Tobs_diff_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'pr') ##
        dsT_ann_diff_delta = csf.compute_predictor_deltas(dsT_ann_diff,dsTobs_ann_diff,'tas')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsT_ann_base_delta, dsT_ann_diff_delta, dsPr_jja_base_delta]

    ## RCM JJA CH (patch: currently test case)
    if cmip == 'RCM' and season_region == 'JJA_CH':
        T_base_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_ch_jja_1971-2020_mean.nc'
        T_diff_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_ch_ann_2011-2020_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_ch_ann_2011-2020_1971-1980_diff.nc'

        # load, filter for common models
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_diff = csf.load_models(path,T_diff_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        Tobs_base_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_jja_1971-2020_mean.nc'
        Tobs_diff_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_ann_2011-2020_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_ann_2011-2020_1971-1980_diff.nc'

        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_diff = csf.load_observations(path,Tobs_diff_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'pr') ##
        dsT_ann_diff_delta = csf.compute_predictor_deltas(dsT_ann_diff,dsTobs_ann_diff,'tas')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsT_ann_base_delta, dsT_ann_diff_delta, dsPr_jja_base_delta]

    ## RCM DJF ALPS
    if cmip == 'RCM' and season_region == 'DJF_ALPS':
        T_base_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_alps_ann_1971-1980_mean.nc'
        T_diff_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_alps_ann_2011-2020_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_alps_djf_1971-2020_mean.nc'

        # load, filter for common models
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_diff = csf.load_models(path,T_diff_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        Tobs_base_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_ann_1971-1980_mean.nc'
        Tobs_diff_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_ann_2011-2020_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_djf_1971-2020_mean.nc'

        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_diff = csf.load_observations(path,Tobs_diff_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_diff_delta = csf.compute_predictor_deltas(dsT_ann_diff,dsTobs_ann_diff,'tas')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsT_ann_base_delta, dsT_ann_diff_delta, dsPr_djf_base_delta]

    ## RCM DJF CH
    if cmip == 'RCM' and season_region == 'DJF_CH':
        T_base_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_ch_ann_1971-1980_mean.nc'
        T_diff_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_ch_ann_2011-2020_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_ch_djf_1971-2020_mean.nc'

        # load, filter for common models
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_diff = csf.load_models(path,T_diff_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        Tobs_base_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_ann_1971-1980_mean.nc'
        Tobs_diff_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_ann_2011-2020_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_djf_1971-2020_mean.nc'

        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_diff = csf.load_observations(path,Tobs_diff_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_diff_delta = csf.compute_predictor_deltas(dsT_ann_diff,dsTobs_ann_diff,'tas')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsT_ann_base_delta, dsT_ann_diff_delta, dsPr_djf_base_delta]

    ## RCM JJA CEU
    if cmip == 'RCM' and season_region == 'JJA_CEU':
        T_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_ceu_jja_1971-2015_trend.nc'

        # load, filter for common models
        dsT_trnd = csf.load_models(path,T_fn,cmip,default_models=default_models)

        # load observations
        Tobs_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ceu_jja_1971-2015_trend.nc'

        dsTobs_trnd = csf.load_observations(path,Tobs_fn)

        # compute predictor-obs RMSEs
        dsT_trnd_delta = csf.compute_predictor_deltas(dsT_trnd,dsTobs_trnd,'tas')

        return [dsT_trnd_delta]

    ## RCM DJF CEU
    if cmip == 'RCM' and season_region == 'DJF_CEU':
        T_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_ceu_djf_1971-2015_trend.nc'

        # load, filter for common models
        dsT_trnd = csf.load_models(path,T_fn,cmip,default_models=default_models)

        # load observations
        Tobs_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ceu_djf_1971-2015_trend.nc'

        dsTobs_trnd = csf.load_observations(path,Tobs_fn)

        # compute predictor-obs RMSEs
        dsT_trnd_delta = csf.compute_predictor_deltas(dsT_trnd,dsTobs_trnd,'tas')

        return [dsT_trnd_delta]

    ##################################################
    # CMIP6 Predictors
    ##################################################

    ## CMIP6 JJA CEU
    if cmip == 'CMIP6' and season_region == 'JJA_CEU':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP6_SSP585_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1950-1969_mean.nc'
        SW_jja_fn = 'swcre_mon_CMIP6_SSP585_g025_CEU_jja_2001-2018_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSW_jja_base = csf.load_models(path,SW_jja_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        SWobs_jja_fn = 'swcre_mon_OBS_g025_CEU_jja_2001-2018_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSWobs_jja_base = csf.load_observations(path,SWobs_jja_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSW_jja_base_delta = csf.compute_predictor_deltas(dsSW_jja_base,dsSWobs_jja_base,'swcre')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSW_jja_base_delta, dsPr_jja_base_delta]

    ## CMIP6 JJA CEU
    if cmip == 'CMIP6' and season_region == 'JJA_CH':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP6_SSP585_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_CH_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_CH_ann_1950-1969_mean.nc'
        SW_jja_fn = 'swcre_mon_CMIP6_SSP585_g025_CH_jja_2001-2018_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_CH_jja_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSW_jja_base = csf.load_models(path,SW_jja_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_CH_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_CH_ann_1950-1969_mean.nc'
        SWobs_jja_fn = 'swcre_mon_OBS_g025_CH_jja_2001-2018_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_CH_jja_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSWobs_jja_base = csf.load_observations(path,SWobs_jja_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSW_jja_base_delta = csf.compute_predictor_deltas(dsSW_jja_base,dsSWobs_jja_base,'swcre')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSW_jja_base_delta, dsPr_jja_base_delta]

    ## CMIP6 DJF NEU
    if cmip == 'CMIP6' and season_region == 'DJF_NEU':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP6_SSP585_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP6_hist_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ## CMIP6 DJF CEU
    if cmip == 'CMIP6' and season_region == 'DJF_CEU':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP6_SSP585_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP6_hist_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_EOBS-CEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-CEU_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ## CMIP6 DJF CH
    if cmip == 'CMIP6' and season_region == 'DJF_CH':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP6_SSP585_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_CH_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_CH_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP6_hist_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_CH_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        SWobs_ann_fn = 'swcre_mon_OBS_g025_SHML_ann_2001-2018_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_CH_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_CH_ann_1950-1969_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_CH_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsSWobs_ann_base = csf.load_observations(path,SWobs_ann_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsSW_ann_base_delta = csf.compute_predictor_deltas(dsSW_ann_base,dsSWobs_ann_base,'swcre')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsSW_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ##################################################
    # CH202x (CMIP6) Predictors
    ##################################################

    ## CH202x_CMIP6 JJA CEU (patch: currently test case)
    if cmip == 'CH202x_CMIP6' and season_region == 'JJA_CEU':
        T_fn = 'tas_mon_CMIP6_SSP585_g025_CEU_jja_1971-2015_trend.nc'

        # load, filter for common models
        dsT_trnd = csf.load_models(path,T_fn,cmip,default_models=default_models)

        # load observations
        Tobs_fn = 'tas_mon_OBS_g025_CEU_jja_1971-2015_trend.nc'
        dsTobs_trnd = csf.load_observations(path,Tobs_fn)

        # compute predictor-obs RMSEs
        dsT_trnd_delta = csf.compute_predictor_deltas(dsT_trnd,dsTobs_trnd,'tas')
        return [dsT_trnd_delta]

    ## CH202x_CMIP6 JJA CH
    if cmip == 'CH202x_CMIP6' and season_region == 'JJA_CH':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1981-2010_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_CH_ann_2001-2010_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_CH_jja_1981-2010_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1981-2010_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_CH_ann_2001-2010_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_OBS_g025_CH_jja_1981-2010_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsPr_jja_base_delta]

    ## CH202x_CMIP6 DJF NEU
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_NEU':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP6_hist_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ## CH202x_CMIP6 DJF CEU (patch: current test case)
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_CEU':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1950-1969_mean.nc'
        R_ann_fn = 'rsds_mon_CMIP6_SSP585_g025_CEU_ann_2001-2018_mean.nc'
        Pr_djf_fn = 'pr_mon_CMIP6_hist_g025_EOBS-CEU_djf_1995-2014_mean.nc'
        Pr_jja_fn = 'pr_mon_CMIP6_hist_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=default_models)
        dsR_ann_base = csf.load_models(path,R_ann_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_djf_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_jja_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        R_ann_fn = 'rsds_mon_OBS_g025_CEU_ann_2001-2018_mean_flip.nc'
        Probs_djf_fn = 'pr_mon_OBS_g025_EOBS-CEU_djf_1995-2014_mean.nc'
        Probs_jja_fn = 'pr_mon_OBS_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsRobs_ann_base = csf.load_observations(path,R_ann_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_djf_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_jja_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsR_ann_base_delta = csf.compute_predictor_deltas(dsR_ann_base,dsRobs_ann_base,'rsds')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsR_ann_base_delta, dsPr_djf_base_delta,dsPr_jja_base_delta]

    ## CH202x_CMIP6 DJF CH
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_CH':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1981-2010_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_CH_ann_2001-2010_1971-1980_diff.nc'
        SLP_djf_fn = 'psl_mon_CMIP6_hist_g025_NATL_djf_1971-2010_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_CH_djf_1981-2010_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=default_models)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1981-2010_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_CH_ann_2001-2010_1971-1980_diff.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1971-2010_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_CH_djf_1981-2010_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ##################################################
    # RCM (CMIP6) Predictors
    ##################################################
    ## RCM_CMIP6 JJA ALPS (no CMIP6 available; patch, test case)
    if cmip == 'RCM_CMIP6' and season_region == 'JJA_ALPS':
        T_base_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_alps_jja_1971-2020_mean.nc'
        T_diff_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_alps_ann_2011-2020_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_alps_ann_2011-2020_1971-1980_diff.nc'

        # load, filter for common models
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_diff = csf.load_models(path,T_diff_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        Tobs_base_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_jja_1971-2020_mean.nc'
        Tobs_diff_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_ann_2011-2020_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_ann_2011-2020_1971-1980_diff.nc'

        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_diff = csf.load_observations(path,Tobs_diff_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'pr') ##
        dsT_ann_diff_delta = csf.compute_predictor_deltas(dsT_ann_diff,dsTobs_ann_diff,'tas')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsT_ann_base_delta, dsT_ann_diff_delta, dsPr_jja_base_delta]

    ## RCM_CMIP6 JJA CH (no CMIP6 available; patch, test case)
    if cmip == 'RCM_CMIP6' and season_region == 'JJA_CH':
        T_base_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_ch_jja_1971-2020_mean.nc'
        T_diff_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_ch_ann_2011-2020_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_ch_ann_2011-2020_1971-1980_diff.nc'

        # load, filter for common models
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_diff = csf.load_models(path,T_diff_fn,cmip,default_models=default_models)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        Tobs_base_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_jja_1971-2020_mean.nc'
        Tobs_diff_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_ann_2011-2020_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_ann_2011-2020_1971-1980_diff.nc'

        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_diff = csf.load_observations(path,Tobs_diff_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'pr') ##
        dsT_ann_diff_delta = csf.compute_predictor_deltas(dsT_ann_diff,dsTobs_ann_diff,'tas')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')

        return [dsT_ann_base_delta, dsT_ann_diff_delta, dsPr_jja_base_delta]

    ## RCM_CMIP6 DJF ALPS (no CMIP6 available; patch)
    if cmip == 'RCM_CMIP6' and season_region == 'DJF_ALPS':
        T_base_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_alps_ann_1971-1980_mean.nc'
        T_diff_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_alps_ann_2011-2020_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_alps_djf_1971-2020_mean.nc'

        # load, filter for common models
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_diff = csf.load_models(path,T_diff_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        Tobs_base_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_ann_1971-1980_mean.nc'
        Tobs_diff_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_ann_2011-2020_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_alps_djf_1971-2020_mean.nc'

        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_diff = csf.load_observations(path,Tobs_diff_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_diff_delta = csf.compute_predictor_deltas(dsT_ann_diff,dsTobs_ann_diff,'tas')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsT_ann_base_delta, dsT_ann_diff_delta, dsPr_djf_base_delta]

    ## RCM_CMIP6 DJF CH (no CMIP6 available; patch)
    if cmip == 'RCM_CMIP6' and season_region == 'DJF_CH':
        T_base_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_ch_ann_1971-1980_mean.nc'
        T_diff_fn = 'tas_mon_EUR-11_rcp85_CH202X_grid_mask_ch_ann_2011-2020_1971-1980_diff.nc'
        Pr_fn = 'pr_mon_EUR-11_rcp85_CH202X_grid_mask_ch_djf_1971-2020_mean.nc'

        # load, filter for common models
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=default_models)
        dsT_ann_diff = csf.load_models(path,T_diff_fn,cmip,default_models=default_models)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=default_models)

        # load observations
        Tobs_base_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_ann_1971-1980_mean.nc'
        Tobs_diff_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_ann_2011-2020_1971-1980_diff.nc'
        Probs_fn = 'pr_mon_0.11deg_rot_v23.1e_remapnn_mask_ch_djf_1971-2020_mean.nc'

        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_diff = csf.load_observations(path,Tobs_diff_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_fn)

        # compute predictor-obs RMSEs
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_diff_delta = csf.compute_predictor_deltas(dsT_ann_diff,dsTobs_ann_diff,'tas')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsT_ann_base_delta, dsT_ann_diff_delta, dsPr_djf_base_delta]

    ## RCM_CMIP6 JJA CEU (no CMIP6 available; patch)
    if cmip == 'RCM_CMIP6' and season_region == 'JJA_CEU':
        T_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_ceu_jja_1971-2015_trend.nc'

        # load, filter for common models
        dsT_trnd = csf.load_models(path,T_fn,cmip,default_models=default_models)

        # load observations
        Tobs_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ceu_jja_1971-2015_trend.nc'

        dsTobs_trnd = csf.load_observations(path,Tobs_fn)

        # compute predictor-obs RMSEs
        dsT_trnd_delta = csf.compute_predictor_deltas(dsT_trnd,dsTobs_trnd,'tas')

        return [dsT_trnd_delta]

    ## RCM_CMIP6 DJF CEU (no CMIP6 available; patch)
    if cmip == 'RCM_CMIP6' and season_region == 'DJF_CEU':
        T_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_ceu_djf_1971-2015_trend.nc'

        # load, filter for common models
        dsT_trnd = csf.load_models(path,T_fn,cmip,default_models=default_models)

        # load observations
        Tobs_fn = 'tas_mon_0.11deg_rot_v23.1e_remapnn_mask_ceu_djf_1971-2015_trend.nc'

        dsTobs_trnd = csf.load_observations(path,Tobs_fn)

        # compute predictor-obs RMSEs
        dsT_trnd_delta = csf.compute_predictor_deltas(dsT_trnd,dsTobs_trnd,'tas')

        return [dsT_trnd_delta]

def pre_process_perf_rest(deltas, deltas_other, cmip, im_or_em, season_region, spread_path,default_models):
    if cmip not in ['CMIP5','CMIP6','CH202x','CH202x_CMIP6','RCM']:
        raise NotImplementedError(cmip)
    if im_or_em not in ['IM','EM']:
        raise NotImplementedError(im_or_em)

    # concatenate to create a CMIP5/6 ensemble for nomalizing
    both_deltas = []

    for d, do in zip(deltas, deltas_other):
        both_deltas.append(xr.concat([d,do],dim='member'))

    # normalize by CMIP5/6 mean
    ds_norm = []

    for delta, both_delta in zip(deltas, both_deltas):
        ds_norm.append(csf.normalize_predictor_deltas(delta,both_delta))

    # compute performance metric
    dsDeltaQ = csf.performance_metric(*ds_norm)

    # aggregate ensemble means or select spread members
    if im_or_em == 'IM':
        return csf.ensemble_mean_or_individual_member(dsDeltaQ,choice='IM',CMIP=cmip,season_region=season_region,spread_path=spread_path,default_models=default_models)
    elif im_or_em == 'EM':
        return csf.ensemble_mean_or_individual_member(dsDeltaQ,choice='EM',CMIP=cmip,season_region=season_region,spread_path=spread_path,default_models=default_models)

# ################################
# Spread
# ################################
# TO DO: add scenarios
def pre_process_spread_load(path, cmip, season_region, default_models):
    # CMIP5 and CMIP6 temperature and precipitation change
    ## CMIP5 JJA
    if cmip == 'CMIP5' and season_region == 'JJA_CEU' :
        changeT_fn = 'tas_CMIP5_rcp85_CEU_jja_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CEU_jja_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP5' and season_region == 'JJA_CH' :
        changeT_fn = 'tas_CMIP5_rcp85_CH_jja_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CH_jja_2041-2060_1995-2014_diff.nc'
    if cmip == 'CH202x' and season_region == 'JJA_CEU' :
        changeT_fn = 'tas_CMIP5_rcp85_CEU_jja_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CEU_jja_2070-2099_1981-2010_diff.nc'
    if cmip == 'CH202x' and season_region == 'JJA_CH' :
        changeT_fn = 'tas_CMIP5_rcp85_CH_jja_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CH_jja_2070-2099_1981-2010_diff.nc'
    if cmip == 'RCM' and season_region == 'JJA_CH' :
        changeT_fn = 'tas_mon_EUR-11_rcp85_ch_jja_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_mon_EUR-11_rcp85_ch_jja_2070-2099_1981-2010_diff.nc'
    if cmip == 'RCM' and season_region == 'JJA_ALPS' :
        changeT_fn = 'tas_mon_EUR-11_rcp85_alps_jja_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_mon_EUR-11_rcp85_alps_jja_2070-2099_1981-2010_diff.nc'
    ## CMIP5 DJF
    if cmip == 'CMIP5' and season_region == 'DJF_NEU' :
        changeT_fn = 'tas_CMIP5_rcp85_NEU_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_NEU_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP5' and season_region == 'DJF_CEU' :
        changeT_fn = 'tas_CMIP5_rcp85_CEU_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CEU_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP5' and season_region == 'DJF_CH' :
        changeT_fn = 'tas_CMIP5_rcp85_CH_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CH_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CH202x' and season_region == 'DJF_NEU' :
        changeT_fn = 'tas_CMIP5_rcp85_NEU_djf_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_NEU_djf_2070-2099_1981-2010_diff.nc'
    if cmip == 'CH202x' and season_region == 'DJF_CEU' :
        changeT_fn = 'tas_CMIP5_rcp85_CEU_djf_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CEU_djf_2070-2099_1981-2010_diff.nc'
    if cmip == 'CH202x' and season_region == 'DJF_CH' :
        changeT_fn = 'tas_CMIP5_rcp85_CH_djf_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CH_djf_2070-2099_1981-2010_diff.nc'
    if cmip == 'RCM' and season_region == 'DJF_CH' :
        changeT_fn = 'tas_mon_EUR-11_rcp85_ch_djf_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_mon_EUR-11_rcp85_ch_djf_2070-2099_1981-2010_diff.nc'
    if cmip == 'RCM' and season_region == 'DJF_ALPS' :
        changeT_fn = 'tas_mon_EUR-11_rcp85_alps_djf_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_mon_EUR-11_rcp85_alps_djf_2070-2099_1981-2010_diff.nc'
    ## CMIP6 JJA
    if cmip == 'CMIP6' and season_region == 'JJA_CEU' :
        changeT_fn = 'tas_CMIP6_SSP585_CEU_jja_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CEU_jja_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP6' and season_region == 'JJA_CH' :
        changeT_fn = 'tas_CMIP6_SSP585_CH_jja_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CH_jja_2041-2060_1995-2014_diff.nc'
    if cmip == 'CH202x_CMIP6' and season_region == 'JJA_CEU' :
        changeT_fn = 'tas_CMIP6_SSP585_CEU_jja_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CEU_jja_2070-2099_1981-2010_diff.nc'
    if cmip == 'CH202x_CMIP6' and season_region == 'JJA_CH' :
        changeT_fn = 'tas_CMIP6_SSP585_CH_jja_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CH_jja_2070-2099_1981-2010_diff.nc'
    ## CMIP6 DJF
    if cmip == 'CMIP6' and season_region == 'DJF_NEU' :
        changeT_fn = 'tas_CMIP6_SSP585_NEU_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_NEU_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP6' and season_region == 'DJF_CEU' :
        changeT_fn = 'tas_CMIP6_SSP585_CEU_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CEU_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP6' and season_region == 'DJF_CH' :
        changeT_fn = 'tas_CMIP6_SSP585_CH_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CH_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_NEU' :
        changeT_fn = 'tas_CMIP6_SSP585_NEU_djf_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_NEU_djf_2070-2099_1981-2010_diff.nc'
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_CEU' :
        changeT_fn = 'tas_CMIP6_SSP585_CEU_djf_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CEU_djf_2070-2099_1981-2010_diff.nc'
    if cmip == 'CH202x_CMIP6' and season_region == 'DJF_CH' :
        changeT_fn = 'tas_CMIP6_SSP585_CH_djf_2070-2099_1981-2010_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CH_djf_2070-2099_1981-2010_diff.nc'

    # load, filter for common models
    dsT_target_ts = csf.load_models(path,changeT_fn,cmip,default_models=default_models)
    dsPr_target_ts = csf.load_models(path,changePr_fn,cmip,default_models=default_models)
    return dsT_target_ts, dsPr_target_ts

def pre_process_spread(path, cmip, im_or_em, season_region,default_models):
    if cmip not in ['CMIP5','CMIP6','CH202x','CH202x_CMIP6','RCM']:
        raise NotImplementedError(cmip)

    dsT_target_ts, dsPr_target_ts = pre_process_spread_load(path, cmip, season_region, default_models=default_models)

    # aggregate ensemble means or select spread members
    dsT_target_ts_sel = csf.ensemble_mean_or_individual_member(dsT_target_ts,choice=im_or_em,CMIP=cmip,season_region=season_region,spread_path=path,key='tas', default_models=default_models)
    dsPr_target_ts_sel = csf.ensemble_mean_or_individual_member(dsPr_target_ts,choice=im_or_em,CMIP=cmip,season_region=season_region,spread_path=path,key='pr', default_models=default_models)
    targets = [dsT_target_ts_sel,dsPr_target_ts_sel]


    # normalize and get squared difference
    ds_sqr_diff = []

    for ds in targets:
        ds_sqr_diff.append(csf.get_squared_diff(csf.normalize_spread_component(ds)))

    # sum and sqrt to compute spread metric
    ds_spread_metric = np.sqrt(ds_sqr_diff[0]+ds_sqr_diff[1])
    return ds_spread_metric, targets

# ################################
# Independence
# ################################
def pre_process_indep_load(path, cmip,default_models):
    # CMIP5 and CMIP6 temperature and precipitation change
    if cmip in ['CMIP5','CH202x']:
        ind_tas_fn = 'tas_mon_CMIP5_hist_g025_indmask_ann_1905-2005_mean.nc'
        ind_psl_fn = 'psl_mon_CMIP5_hist_g025_indmask_ann_1905-2005_mean.nc'
    if cmip == 'RCM':
        ind_tas_fn = 'tas_mon_EUR-11_hist_CH202X_grid_mask_ann_1971-2005_mean.nc'
        ind_psl_fn = 'psl_mon_EUR-11_hist_CH202X_grid_mask_ann_1971-2005_mean.nc'
    if cmip in ['CMIP6','CH202x_CMIP6']:
        ind_tas_fn = 'tas_mon_CMIP6_hist_g025_indmask_ann_1905-2005_mean.nc'
        ind_psl_fn = 'psl_mon_CMIP6_hist_g025_indmask_ann_1905-2005_mean.nc'

    # load, filter for common models
    dsT_clim_mask = csf.load_models(path,ind_tas_fn,cmip,default_models=default_models)
    dsP_clim_mask = csf.load_models(path,ind_psl_fn,cmip,default_models=default_models)
    return dsT_clim_mask, dsP_clim_mask

def pre_process_indep(path, cmip, im_or_em,season_region,spread_path, default_models):
    if cmip not in ['CMIP5','CMIP6','CH202x','CH202x_CMIP6','RCM']:
        raise NotImplementedError(cmip)

    dsT_clim_mask, dsP_clim_mask = pre_process_indep_load(path, cmip,default_models=default_models)

    if cmip in ['CMIP5','CMIP6','CH202x','CH202x_CMIP6']:
        # aggregate ensemble means or select spread members
        dsT_clim_mask_sel = csf.ensemble_mean_or_individual_member(dsT_clim_mask,choice=im_or_em,CMIP=cmip,season_region=season_region,spread_path=spread_path,key='tas', default_models=default_models)
        dsP_clim_mask_sel = csf.ensemble_mean_or_individual_member(dsP_clim_mask,choice=im_or_em,CMIP=cmip,season_region=season_region,spread_path=spread_path,key='psl', default_models=default_models)
        inds = [dsT_clim_mask_sel,dsP_clim_mask_sel]

        # get and normailize inter-model RMSEs
        ds_clim_mask_err = []

        for ds in inds:
            ds_clim_mask_err.append(csf.normalize_independence_matrix(csf.get_error(ds)))

        # average to compute independence metric
        dsWi = (ds_clim_mask_err[0]+ds_clim_mask_err[1])/len(ds_clim_mask_err)

    if cmip == 'RCM':
        # aggregate ensemble means or select spread members
        dsT_clim_mask_sel = csf.ensemble_mean_or_individual_member(dsT_clim_mask,choice=im_or_em,CMIP=cmip,season_region=season_region,spread_path=spread_path,key='tas', default_models=default_models)
        dsP_clim_mask_sel = csf.ensemble_mean_or_individual_member(dsP_clim_mask,choice=im_or_em,CMIP=cmip,season_region=season_region,spread_path=spread_path,key='pr', default_models=default_models)
        inds = [dsT_clim_mask_sel,dsP_clim_mask_sel]

        # get and normailize inter-model RMSEs
        ds_clim_mask_err = []

        for ds in inds:
            ds_clim_mask_err.append(csf.normalize_independence_matrix(csf.get_error(ds)))

        # average to compute independence metric
        dsWi = (ds_clim_mask_err[0]+ds_clim_mask_err[1])/len(ds_clim_mask_err)
    return dsWi
