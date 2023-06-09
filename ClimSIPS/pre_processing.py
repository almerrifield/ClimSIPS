import xarray as xr
import numpy as np

from . import function as csf

# ################################
# Performance
# ################################


def pre_process_perf(path, cmip, im_or_em, season_region):
    deltas_5 = pre_process_perf_load_delta(path, 'CMIP5', season_region)
    deltas_6 = pre_process_perf_load_delta(path, 'CMIP6', season_region)
    return pre_process_perf_rest(deltas_5, deltas_6, cmip, im_or_em, season_region)

def pre_process_perf_load_delta(path, cmip, season_region):
    if cmip not in ['CMIP5','CMIP6']:
        raise NotImplementedError(cmip)

    # CMIP5 and CMIP6 predictors
    ## CMIP5 JJA CEU
    if cmip == 'CMIP5' and season_region == 'JJA_CEU':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP5_rcp85_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_EUR_ann_1950-1969_mean.nc'
        SW_jja_fn = 'swcre_mon_CMIP5_rcp85_g025_CEU_jja_2001-2018_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=True)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=True)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=True)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=True)
        dsSW_jja_base = csf.load_models(path,SW_jja_fn,cmip,default_models=True)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=True)

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

    ## CMIP5 DJF NEU
    if cmip == 'CMIP5' and season_region == 'DJF_NEU':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP5_rcp85_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_EUR_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP5_rcp85_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=True)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=True)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=True)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=True)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=True)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=True)

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
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=True)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=True)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=True)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=True)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=True)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=True)

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

    ## CMIP5 CH202x Case
    if cmip == 'CMIP5' and season_region == 'CH202x_CEU':
        SST_fn = 'tos_mon_CMIP5_rcp85_g025_NAWH_ann_1995-2014_mean.nc'
        T_base_fn = 'tas_mon_CMIP5_rcp85_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP5_hist_g025_EUR_ann_1950-1969_mean.nc'
        Pr_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-CEU_jja_1995-2014_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP5_rcp85_g025_NATL_djf_1950-2014_mean.nc'
        Pr_djf_fn = 'pr_mon_CMIP5_rcp85_g025_EOBS-CEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=True)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=True)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=True)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=True)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=True)
        dsPr_djf_base = csf.load_models(path,Pr_djf_fn,cmip,default_models=True)

        # load observations
        SSTobs_fn = 'tos_mon_OBS_g025_NAWH_ann_1995-2014_mean.nc'
        Tobs_base_fn = 'tas_mon_OBS_g025_EUR_ann_1995-2014_mean.nc'
        Tobs_his_fn = 'tas_mon_OBS_g025_EUR_ann_1950-1969_mean.nc'
        Probs_fn = 'pr_mon_OBS_g025_EOBS-CEU_jja_1995-2014_mean.nc'
        SLPobs_djf_fn = 'psl_mon_OBS_g025_NATL_djf_1950-2014_mean.nc'
        Probs_djf_fn = 'pr_mon_OBS_g025_EOBS-CEU_djf_1995-2014_mean.nc'

        dsSSTobs_ann_base = csf.load_observations(path,SSTobs_fn)
        dsTobs_ann_base = csf.load_observations(path,Tobs_base_fn)
        dsTobs_ann_his = csf.load_observations(path,Tobs_his_fn)
        dsProbs_jja_base = csf.load_observations(path,Probs_fn)
        dsSLPobs_djf_base = csf.load_observations(path,SLPobs_djf_fn)
        dsProbs_djf_base = csf.load_observations(path,Probs_djf_fn)

        # compute predictor-obs RMSEs
        dsSST_ann_base_delta = csf.compute_predictor_deltas(dsSST_ann_base,dsSSTobs_ann_base,'tos')
        dsT_ann_base_delta = csf.compute_predictor_deltas(dsT_ann_base,dsTobs_ann_base,'tas')
        dsT_ann_his_delta = csf.compute_predictor_deltas(dsT_ann_his,dsTobs_ann_his,'tas')
        dsPr_jja_base_delta = csf.compute_predictor_deltas(dsPr_jja_base,dsProbs_jja_base,'pr')
        dsSLP_djf_base_delta = csf.compute_predictor_deltas(dsSLP_djf_base,dsSLPobs_djf_base,'psl')
        dsPr_djf_base_delta = csf.compute_predictor_deltas(dsPr_djf_base,dsProbs_djf_base,'pr')

        return [dsSST_ann_base_delta, dsT_ann_base_delta, dsT_ann_his_delta, dsPr_jja_base_delta, dsSLP_djf_base_delta, dsPr_djf_base_delta]

    ## CMIP6 JJA CEU
    if cmip == 'CMIP6' and season_region == 'JJA_CEU':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP6_SSP585_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1950-1969_mean.nc'
        SW_jja_fn = 'swcre_mon_CMIP6_SSP585_g025_CEU_jja_2001-2018_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_EOBS-CEU_jja_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=True)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=True)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=True)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=True)
        dsSW_jja_base = csf.load_models(path,SW_jja_fn,cmip,default_models=True)
        dsPr_jja_base = csf.load_models(path,Pr_fn,cmip,default_models=True)

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

    ## CMIP6 DJF NEU
    if cmip == 'CMIP6' and season_region == 'DJF_NEU':
        SST_fn = 'tos_mon_CMIP6_hist_g025_NAWH_ann_1995-2014_mean.nc'
        SW_ann_fn = 'swcre_mon_CMIP6_SSP585_g025_SHML_ann_2001-2018_mean.nc'
        T_base_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1995-2014_mean.nc'
        T_his_fn = 'tas_mon_CMIP6_hist_g025_EUR_ann_1950-1969_mean.nc'
        SLP_djf_fn = 'psl_mon_CMIP6_hist_g025_NATL_djf_1950-2014_mean.nc'
        Pr_fn = 'pr_mon_CMIP6_hist_g025_EOBS-NEU_djf_1995-2014_mean.nc'

        # load, filter for common models
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=True)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=True)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=True)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=True)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=True)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=True)

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
        dsSST_ann_base = csf.load_models(path,SST_fn,cmip,default_models=True)
        dsSW_ann_base = csf.load_models(path,SW_ann_fn,cmip,default_models=True)
        dsT_ann_base = csf.load_models(path,T_base_fn,cmip,default_models=True)
        dsT_ann_his = csf.load_models(path,T_his_fn,cmip,default_models=True)
        dsSLP_djf_base = csf.load_models(path,SLP_djf_fn,cmip,default_models=True)
        dsPr_djf_base = csf.load_models(path,Pr_fn,cmip,default_models=True)

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


def pre_process_perf_rest(deltas_5, deltas_6, cmip, im_or_em, season_region):
    if cmip not in ['CMIP5','CMIP6']:
        raise NotImplementedError(cmip)
    if im_or_em not in ['IM','EM']:
        raise NotImplementedError(im_or_em)

    # concatenate to create a CMIP5/6 ensemble for nomalizing
    both_deltas = []

    for d5, d6 in zip(deltas_5, deltas_6):
        both_deltas.append(xr.concat([d5,d6],dim='member'))

    if cmip == 'CMIP5':
        deltas = deltas_5
    elif cmip == 'CMIP6':
        deltas = deltas_6

    # normalize by CMIP5/6 mean
    ds_norm = []

    for delta, both_delta in zip(deltas, both_deltas):
        ds_norm.append(csf.normalize_predictor_deltas(delta,both_delta))

    # compute performance metric
    dsDeltaQ = csf.performance_metric(*ds_norm)

    # aggregate ensemble means or select spread members
    if im_or_em == 'IM':
        return csf.ensemble_mean_or_individual_member(dsDeltaQ,choice='IM',CMIP=cmip,season_region=season_region)
    elif im_or_em == 'EM':
        return csf.ensemble_mean_or_individual_member(dsDeltaQ,choice='EM',CMIP=cmip,season_region=season_region)

# ################################
# Spread
# ################################

def pre_process_spread(path, cmip, im_or_em, season_region):
    if cmip not in ['CMIP5','CMIP6']:
        raise NotImplementedError(cmip)

    # CMIP5 and CMIP6 temperature and precipitation change
    if cmip == 'CMIP5' and season_region == 'JJA_CEU' :
        changeT_fn = 'tas_CMIP5_rcp85_CEU_jja_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CEU_jja_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP5' and season_region == 'DJF_NEU' :
        changeT_fn = 'tas_CMIP5_rcp85_NEU_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_NEU_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP5' and season_region == 'DJF_CEU' :
        changeT_fn = 'tas_CMIP5_rcp85_CEU_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CEU_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP5' and season_region == 'CH202x_CEU' :
        changeT_fn = 'tas_CMIP5_rcp85_CEU_ann_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP5_rcp85_CEU_ann_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP6' and season_region == 'JJA_CEU' :
        changeT_fn = 'tas_CMIP6_SSP585_CEU_jja_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CEU_jja_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP6' and season_region == 'DJF_NEU' :
        changeT_fn = 'tas_CMIP6_SSP585_NEU_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_NEU_djf_2041-2060_1995-2014_diff.nc'
    if cmip == 'CMIP6' and season_region == 'DJF_CEU' :
        changeT_fn = 'tas_CMIP6_SSP585_CEU_djf_2041-2060_1995-2014_diff.nc'
        changePr_fn = 'pr_CMIP6_SSP585_CEU_djf_2041-2060_1995-2014_diff.nc'

    # load, filter for common models
    dsT_target_ts = csf.load_models(path,changeT_fn,cmip,default_models=True)
    dsPr_target_ts = csf.load_models(path,changePr_fn,cmip,default_models=True)

    # aggregate ensemble means or select spread members
    dsT_target_ts_sel = csf.ensemble_mean_or_individual_member(dsT_target_ts,choice=im_or_em,CMIP=cmip,season_region=season_region,key='tas')
    dsPr_target_ts_sel = csf.ensemble_mean_or_individual_member(dsPr_target_ts,choice=im_or_em,CMIP=cmip,season_region=season_region,key='pr')
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

def pre_process_indep(path, cmip, im_or_em,season_region):
    if cmip not in ['CMIP5','CMIP6']:
        raise NotImplementedError(cmip)

    # CMIP5 and CMIP6 temperature and precipitation change
    if cmip == 'CMIP5':
        ind_tas_fn = 'tas_mon_CMIP5_hist_g025_indmask_ann_1905-2005_mean.nc'
        ind_psl_fn = 'psl_mon_CMIP5_hist_g025_indmask_ann_1905-2005_mean.nc'
    if cmip == 'CMIP6':
        ind_tas_fn = 'tas_mon_CMIP6_hist_g025_indmask_ann_1905-2005_mean.nc'
        ind_psl_fn = 'psl_mon_CMIP6_hist_g025_indmask_ann_1905-2005_mean.nc'

    # load, filter for common models
    dsT_clim_mask = csf.load_models(path,ind_tas_fn,cmip,default_models=True)
    dsP_clim_mask = csf.load_models(path,ind_psl_fn,cmip,default_models=True)

    # aggregate ensemble means or select spread members
    dsT_clim_mask_sel = csf.ensemble_mean_or_individual_member(dsT_clim_mask,choice=im_or_em,CMIP=cmip,season_region=season_region,key='tas')
    dsP_clim_mask_sel = csf.ensemble_mean_or_individual_member(dsP_clim_mask,choice=im_or_em,CMIP=cmip,season_region=season_region,key='psl')
    inds = [dsT_clim_mask_sel,dsP_clim_mask_sel]

    # get and normailize inter-model RMSEs
    ds_clim_mask_err = []

    for ds in inds:
        ds_clim_mask_err.append(csf.normalize_independence_matrix(csf.get_error(ds)))

    # average to compute independence metric
    dsWi = (ds_clim_mask_err[0]+ds_clim_mask_err[1])/len(ds_clim_mask_err)
    return dsWi
