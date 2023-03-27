#################################
# packages
#################################

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import datetime
from scipy import signal, stats
from statsmodels.stats.weightstats import DescrStatsW
import xskillscore

import itertools
import math
import time
import random
import sys
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from . import member_selection as csms

##################################################################
# functions for output file creations
##################################################################

# select default common models
# for user defined sets, see member_selection.py
def select_default_common_models(ds,CMIP):
    if CMIP == 'CMIP6':
        members = csms.CMIP6_common_members
    if CMIP == 'CMIP5':
        members = csms.CMIP5_common_members
    return ds.sel(member=members)

# load performance predictors
def load_models(path,filename,CMIP,default_models=True):
    res = xr.open_dataset(path+filename,use_cftime = True)
    res = res.sortby(res.member)
    if default_models:
        res = select_default_common_models(res,CMIP)
    return res

# load observations
def load_observations(path,filename):
    res = xr.open_dataset(path+filename,use_cftime = True)
    return res

# cosine-latitude weighted average
def cos_lat_weighted_mean(ds):
  weights = np.cos(np.deg2rad(ds.lat))
  weights.name = "weights"
  ds_weighted = ds.weighted(weights)
  weighted_mean = ds_weighted.mean(('lon', 'lat'))
  return weighted_mean

# compute rmse between model and observations
def compute_predictor_deltas(ds,ds_obs,key):
    weights = [np.cos(np.deg2rad(ds.lat))]*len(ds.lon)
    weights = xr.concat(weights, "lon")
    weights['lon'] = ds.lon
    return xskillscore.rmse(ds[key],ds_obs[key],dim=['lat','lon'],weights=weights,skipna=True)

# normalize performance predictors
def normalize_predictor_deltas(ds,ds_cat):
    ds_norm = ds/ds_cat.mean('member')
    return ds_norm

# compute performace metricI
# need to be able to compute for an arbitrary number of predictors
def performance_metric(*ds_list):
    return sum(ds_list)/len(ds_list)

# choose ensemble mean or individual member
# TO DO: generalize ensemble mean for any base select
# TO DO: provide way to select the spread-i-est members, generalize
def ensemble_mean_or_individual_member(ds,choice,CMIP,season_region,key=None):
    if key:
        dss = ds[key]
    else:
        dss = ds
    ## CMIP6 by ensemble mean
    if choice == 'EM' and CMIP == 'CMIP6':
        hadgemmm = dss.sel(member = ['HadGEM3-GC31-MM-r1i1p1f3',
        'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
        'HadGEM3-GC31-MM-r4i1p1f3']).mean('member')
        hadgemmm['member'] = 'HadGEM3-GC31-MM-r0i0p0f0'
        mpihr = dss.sel(member=['MPI-ESM1-2-HR-r1i1p1f1',
        'MPI-ESM1-2-HR-r2i1p1f1']).mean('member')
        mpihr['member'] = 'MPI-ESM1-2-HR-r0i0p0f0'
        mri2 = dss.sel(member=['MRI-ESM2-0-r1i1p1f1', 'MRI-ESM2-0-r1i2p1f1']).mean('member')
        mri2['member'] = 'MRI-ESM2-0-r0i0p0f0'
        cesm2_waccm = dss.sel(member=['CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1', 'CESM2-WACCM-r3i1p1f1']).mean('member')
        cesm2_waccm['member'] = 'CESM2-WACCM-r0i0p0f0'
        cesm2 = dss.sel(member=['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1', 'CESM2-r1i1p1f1',
        'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1']).mean('member')
        cesm2['member'] = 'CESM2-r0i0p0f0'
        cnrm2 = dss.sel(member=['CNRM-ESM2-1-r1i1p1f2',
        'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2', 'CNRM-ESM2-1-r4i1p1f2',
        'CNRM-ESM2-1-r5i1p1f2']).mean('member')
        cnrm2['member'] = 'CNRM-ESM2-1-r0i0p0f0'
        cnrm6 = dss.sel(member=['CNRM-CM6-1-r1i1p1f2',
        'CNRM-CM6-1-r2i1p1f2', 'CNRM-CM6-1-r3i1p1f2', 'CNRM-CM6-1-r4i1p1f2',
        'CNRM-CM6-1-r5i1p1f2', 'CNRM-CM6-1-r6i1p1f2']).mean('member')
        cnrm6['member'] = 'CNRM-CM6-1-r0i0p0f0'
        hadgemll = dss.sel(member=['HadGEM3-GC31-LL-r1i1p1f3',
        'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
        'HadGEM3-GC31-LL-r4i1p1f3']).mean('member')
        hadgemll['member'] = 'HadGEM3-GC31-LL-r0i0p0f0'
        access_cm2 = dss.sel(member=['ACCESS-CM2-r1i1p1f1', 'ACCESS-CM2-r2i1p1f1', 'ACCESS-CM2-r3i1p1f1']).mean('member')
        access_cm2['member'] = 'ACCESS-CM2-r0i0p0f0'
        ipsl6a = dss.sel(member=['IPSL-CM6A-LR-r14i1p1f1', 'IPSL-CM6A-LR-r1i1p1f1',
        'IPSL-CM6A-LR-r2i1p1f1', 'IPSL-CM6A-LR-r3i1p1f1',
        'IPSL-CM6A-LR-r4i1p1f1', 'IPSL-CM6A-LR-r6i1p1f1']).mean('member')
        ipsl6a['member'] = 'IPSL-CM6A-LR-r0i0p0f0'
        access_5 = dss.sel(member=['ACCESS-ESM1-5-r10i1p1f1', 'ACCESS-ESM1-5-r1i1p1f1',
        'ACCESS-ESM1-5-r2i1p1f1', 'ACCESS-ESM1-5-r3i1p1f1',
        'ACCESS-ESM1-5-r4i1p1f1', 'ACCESS-ESM1-5-r5i1p1f1',
        'ACCESS-ESM1-5-r6i1p1f1', 'ACCESS-ESM1-5-r7i1p1f1',
        'ACCESS-ESM1-5-r8i1p1f1', 'ACCESS-ESM1-5-r9i1p1f1']).mean('member')
        access_5['member'] = 'ACCESS-ESM1-5-r0i0p0f0'
        uk = dss.sel(member=['UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
        'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2', 'UKESM1-0-LL-r8i1p1f2']).mean('member')
        uk['member'] = 'UKESM1-0-LL-r0i0p0f0'
        mpi2lr = dss.sel(member=['MPI-ESM1-2-LR-r10i1p1f1',
        'MPI-ESM1-2-LR-r1i1p1f1', 'MPI-ESM1-2-LR-r2i1p1f1',
        'MPI-ESM1-2-LR-r3i1p1f1', 'MPI-ESM1-2-LR-r4i1p1f1',
        'MPI-ESM1-2-LR-r5i1p1f1', 'MPI-ESM1-2-LR-r6i1p1f1',
        'MPI-ESM1-2-LR-r7i1p1f1', 'MPI-ESM1-2-LR-r8i1p1f1',
        'MPI-ESM1-2-LR-r9i1p1f1']).mean('member')
        mpi2lr['member'] = 'MPI-ESM1-2-LR-r0i0p0f0'
        canesm5 = dss.sel(member=['CanESM5-r10i1p1f1', 'CanESM5-r10i1p2f1',
        'CanESM5-r11i1p1f1', 'CanESM5-r11i1p2f1', 'CanESM5-r12i1p1f1',
        'CanESM5-r12i1p2f1', 'CanESM5-r13i1p1f1', 'CanESM5-r13i1p2f1',
        'CanESM5-r14i1p1f1', 'CanESM5-r14i1p2f1', 'CanESM5-r15i1p1f1',
        'CanESM5-r15i1p2f1', 'CanESM5-r16i1p1f1', 'CanESM5-r16i1p2f1',
        'CanESM5-r17i1p1f1', 'CanESM5-r17i1p2f1', 'CanESM5-r18i1p1f1',
        'CanESM5-r18i1p2f1', 'CanESM5-r19i1p1f1', 'CanESM5-r19i1p2f1',
        'CanESM5-r1i1p1f1', 'CanESM5-r1i1p2f1', 'CanESM5-r20i1p1f1',
        'CanESM5-r20i1p2f1', 'CanESM5-r21i1p1f1', 'CanESM5-r21i1p2f1',
        'CanESM5-r22i1p1f1', 'CanESM5-r22i1p2f1', 'CanESM5-r23i1p1f1',
        'CanESM5-r23i1p2f1', 'CanESM5-r24i1p1f1', 'CanESM5-r24i1p2f1',
        'CanESM5-r25i1p1f1', 'CanESM5-r25i1p2f1', 'CanESM5-r2i1p1f1',
        'CanESM5-r2i1p2f1', 'CanESM5-r3i1p1f1', 'CanESM5-r3i1p2f1',
        'CanESM5-r4i1p1f1', 'CanESM5-r4i1p2f1', 'CanESM5-r5i1p1f1',
        'CanESM5-r5i1p2f1', 'CanESM5-r6i1p1f1', 'CanESM5-r6i1p2f1',
        'CanESM5-r7i1p1f1', 'CanESM5-r7i1p2f1', 'CanESM5-r8i1p1f1',
        'CanESM5-r8i1p2f1', 'CanESM5-r9i1p1f1', 'CanESM5-r9i1p2f1']).mean('member')
        canesm5['member'] = 'CanESM5-r0i0p0f0'
        miroce = dss.sel(member=['MIROC-ES2L-r10i1p1f2',
        'MIROC-ES2L-r1i1p1f2', 'MIROC-ES2L-r2i1p1f2', 'MIROC-ES2L-r3i1p1f2',
        'MIROC-ES2L-r4i1p1f2', 'MIROC-ES2L-r5i1p1f2', 'MIROC-ES2L-r6i1p1f2',
        'MIROC-ES2L-r7i1p1f2', 'MIROC-ES2L-r8i1p1f2', 'MIROC-ES2L-r9i1p1f2']).mean('member')
        miroce['member'] = 'MIROC-ES2L-r0i0p0f0'
        miroc6 = dss.sel(member=['MIROC6-r10i1p1f1', 'MIROC6-r11i1p1f1', 'MIROC6-r12i1p1f1',
        'MIROC6-r13i1p1f1', 'MIROC6-r14i1p1f1', 'MIROC6-r15i1p1f1',
        'MIROC6-r16i1p1f1', 'MIROC6-r17i1p1f1', 'MIROC6-r18i1p1f1',
        'MIROC6-r19i1p1f1', 'MIROC6-r1i1p1f1', 'MIROC6-r20i1p1f1',
        'MIROC6-r21i1p1f1', 'MIROC6-r22i1p1f1', 'MIROC6-r23i1p1f1',
        'MIROC6-r24i1p1f1', 'MIROC6-r25i1p1f1', 'MIROC6-r26i1p1f1',
        'MIROC6-r27i1p1f1', 'MIROC6-r28i1p1f1', 'MIROC6-r29i1p1f1',
        'MIROC6-r2i1p1f1', 'MIROC6-r30i1p1f1', 'MIROC6-r31i1p1f1',
        'MIROC6-r32i1p1f1', 'MIROC6-r33i1p1f1', 'MIROC6-r34i1p1f1',
        'MIROC6-r35i1p1f1', 'MIROC6-r36i1p1f1', 'MIROC6-r37i1p1f1',
        'MIROC6-r38i1p1f1', 'MIROC6-r39i1p1f1', 'MIROC6-r3i1p1f1',
        'MIROC6-r40i1p1f1', 'MIROC6-r41i1p1f1', 'MIROC6-r42i1p1f1',
        'MIROC6-r43i1p1f1', 'MIROC6-r44i1p1f1', 'MIROC6-r45i1p1f1',
        'MIROC6-r46i1p1f1', 'MIROC6-r47i1p1f1', 'MIROC6-r48i1p1f1',
        'MIROC6-r49i1p1f1', 'MIROC6-r4i1p1f1', 'MIROC6-r50i1p1f1',
        'MIROC6-r5i1p1f1', 'MIROC6-r6i1p1f1', 'MIROC6-r7i1p1f1',
        'MIROC6-r8i1p1f1', 'MIROC6-r9i1p1f1']).mean('member')
        miroc6['member'] = 'MIROC6-r0i0p0f0'
        nesm3 = dss.sel(member=['NESM3-r1i1p1f1', 'NESM3-r2i1p1f1']).mean('member')
        nesm3['member'] = 'NESM3-r0i0p0f0'
        fgoalsg = dss.sel(member=['FGOALS-g3-r1i1p1f1',
        'FGOALS-g3-r2i1p1f1']).mean('member')
        fgoalsg['member'] = 'FGOALS-g3-r0i0p0f0'
        kace = dss.sel(member=['KACE-1-0-G-r2i1p1f1',
        'KACE-1-0-G-r3i1p1f1']).mean('member')
        kace['member'] = 'KACE-1-0-G-r0i0p0f0'
        cas = dss.sel(member=['CAS-ESM2-0-r1i1p1f1', 'CAS-ESM2-0-r3i1p1f1']).mean('member')
        cas['member'] = 'CAS-ESM2-0-r0i0p0f0'
        ds_all = xr.concat([access_cm2,access_5,dss.sel(member='AWI-CM-1-1-MR-r1i1p1f1'),cas,
        cesm2_waccm,cesm2,dss.sel(member='CMCC-CM2-SR5-r1i1p1f1'),dss.sel(member='CMCC-ESM2-r1i1p1f1'),
        dss.sel(member='CNRM-CM6-1-HR-r1i1p1f2'),cnrm6,cnrm2,canesm5,dss.sel(member='E3SM-1-1-r1i1p1f1'),
        dss.sel(member='FGOALS-f3-L-r1i1p1f1'),fgoalsg,dss.sel(member='GFDL-CM4-r1i1p1f1'),
        dss.sel(member='GFDL-ESM4-r1i1p1f1'),dss.sel(member='GISS-E2-1-G-r1i1p3f1'),hadgemll,
        hadgemmm,dss.sel(member='INM-CM4-8-r1i1p1f1'),dss.sel(member='INM-CM5-0-r1i1p1f1'),
        ipsl6a,kace,dss.sel(member='KIOST-ESM-r1i1p1f1'),miroce,miroc6,mpihr,mpi2lr,mri2,nesm3,
        dss.sel(member='NorESM2-MM-r1i1p1f1'),dss.sel(member='TaiESM1-r1i1p1f1'),uk],dim='member')
        return ds_all
# CMIP5 by ensemble mean
    if choice == 'EM' and CMIP == 'CMIP5':
        cesm1 = dss.sel(member=['CESM1-CAM5-r1i1p1', 'CESM1-CAM5-r2i1p1', 'CESM1-CAM5-r3i1p1']).mean('member')
        cesm1['member'] = 'CESM1-CAM5-r0i0p0'
        miroc5 = dss.sel(member=['MIROC5-r1i1p1', 'MIROC5-r2i1p1', 'MIROC5-r3i1p1']).mean('member')
        miroc5['member'] = 'MIROC5-r0i0p0'
        hadgemes = dss.sel(member=['HadGEM2-ES-r1i1p1','HadGEM2-ES-r2i1p1', 'HadGEM2-ES-r3i1p1', 'HadGEM2-ES-r4i1p1']).mean('member')
        hadgemes['member'] = 'HadGEM2-ES-r0i0p0'
        ccsm4 = dss.sel(member=['CCSM4-r1i1p1', 'CCSM4-r2i1p1','CCSM4-r3i1p1', 'CCSM4-r4i1p1', 'CCSM4-r5i1p1', 'CCSM4-r6i1p1']).mean('member')
        ccsm4['member'] = 'CCSM4-r0i0p0'
        canesm2 = dss.sel(member=['CanESM2-r1i1p1', 'CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1','CanESM2-r5i1p1']).mean('member')
        canesm2['member'] = 'CanESM2-r0i0p0'
        mpilr = dss.sel(member=['CanESM2-r1i1p1', 'CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1',
        'CanESM2-r5i1p1']).mean('member')
        mpilr['member'] = 'MPI-ESM-LR-r0i0p0'
        csiro = dss.sel(member=['CSIRO-Mk3-6-0-r10i1p1',
        'CSIRO-Mk3-6-0-r1i1p1', 'CSIRO-Mk3-6-0-r2i1p1', 'CSIRO-Mk3-6-0-r3i1p1',
        'CSIRO-Mk3-6-0-r4i1p1', 'CSIRO-Mk3-6-0-r5i1p1', 'CSIRO-Mk3-6-0-r6i1p1',
        'CSIRO-Mk3-6-0-r7i1p1', 'CSIRO-Mk3-6-0-r8i1p1', 'CSIRO-Mk3-6-0-r9i1p1']).mean('member')
        csiro['member'] = 'CSIRO-Mk3-6-0-r0i0p0'
        gissr = dss.sel(member=['GISS-E2-R-r1i1p1', 'GISS-E2-R-r1i1p2', 'GISS-E2-R-r1i1p3',
        'GISS-E2-R-r2i1p1', 'GISS-E2-R-r2i1p3']).mean('member')
        gissr['member'] = 'GISS-E2-R-r0i0p0'
        gissh = dss.sel(member=['GISS-E2-H-r1i1p1', 'GISS-E2-H-r1i1p2',
        'GISS-E2-H-r1i1p3', 'GISS-E2-H-r2i1p1', 'GISS-E2-H-r2i1p3']).mean('member')
        gissh['member'] = 'GISS-E2-H-r0i0p0'
        cnrm5 = dss.sel(member=['CNRM-CM5-r10i1p1', 'CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1',
        'CNRM-CM5-r4i1p1', 'CNRM-CM5-r6i1p1']).mean('member')
        cnrm5['member'] = 'CNRM-CM5-r0i0p0'
        ipsl5a = dss.sel(member=['IPSL-CM5A-LR-r1i1p1', 'IPSL-CM5A-LR-r2i1p1', 'IPSL-CM5A-LR-r3i1p1',
        'IPSL-CM5A-LR-r4i1p1']).mean('member')
        ipsl5a['member'] = 'IPSL-CM5A-LR-r0i0p0'
        ds_all = xr.concat([dss.sel(member='ACCESS1-0-r1i1p1'),
        dss.sel(member='ACCESS1-3-r1i1p1'),ccsm4,cesm1,cnrm5,csiro,canesm2,
        dss.sel(member='GFDL-CM3-r1i1p1'),dss.sel(member='GFDL-ESM2G-r1i1p1'),
        dss.sel(member='GFDL-ESM2M-r1i1p1'),gissh,gissr,hadgemes,ipsl5a,
        dss.sel(member='IPSL-CM5A-MR-r1i1p1'),dss.sel(member='IPSL-CM5B-LR-r1i1p1'),
        dss.sel(member='MIROC-ESM-r1i1p1'),miroc5,mpilr,dss.sel(member='MPI-ESM-MR-r1i1p1'),
        dss.sel(member='MRI-CGCM3-r1i1p1'),dss.sel(member='NorESM1-M-r1i1p1'),
        dss.sel(member='NorESM1-ME-r1i1p1'),dss.sel(member='bcc-csm1-1-m-r1i1p1'),
        dss.sel(member='bcc-csm1-1-r1i1p1'),dss.sel(member='inmcm4-r1i1p1')],dim='member')
        return ds_all
# CMIP6 by spread
    if choice == 'IM' and CMIP == 'CMIP6':
        mem_out = csms.CMIP6_spread_maximizing_members(csms.CMIP6_common_members,season_region)
        dss = dss.sel(member=mem_out)
        return dss.sortby(dss.member)
# CMIP5 by spread
    if choice == 'IM' and CMIP == 'CMIP5':
        mem_out = csms.CMIP5_spread_maximizing_members(csms.CMIP5_common_members,season_region)
        dss = dss.sel(member=mem_out)
        return dss.sortby(dss.member)

# normalize for spread
def normalize_spread_component(ds):
    return (ds  - np.mean(ds))/np.std(ds)

# compute squared difference for spread metric
def get_squared_diff(ds):
    mod_coords = ds.member.values
    nmod = len(mod_coords)
    res = xr.DataArray(np.empty(shape=(nmod, nmod)),
                        dims=("member", "member_model"), coords=dict(member=mod_coords, member_model=mod_coords))

    for mod1 in ds.transpose("member", ...):
        for mod2 in ds.transpose("member", ...):
            a = (mod1-mod2)**2
            res.loc[dict(member=mod1.member, member_model=mod2.member)] = a
    return res.where(res!=0)


# compute independence matrix
def get_error(ds):
    weights = [np.cos(np.deg2rad(ds.lat))]*len(ds.lon)
    weights = xr.concat(weights, "lon")
    weights['lon'] = ds.lon
    mod_coords = ds.member.values
    nmod = len(mod_coords)
    res = xr.DataArray(np.empty(shape=(nmod, nmod)),
                        dims=("member", "member_model"), coords=dict(member=mod_coords, member_model=mod_coords))

    for mod1 in ds.transpose("member", ...):
        for mod2 in ds.transpose("member", ...):
            a = xskillscore.rmse(mod1,mod2,dim=['lat','lon'],weights=weights,skipna=True)
            res.loc[dict(member=mod1.member, member_model=mod2.member)] = a

    return res.where(res!=0)

# normalize for independence
def normalize_independence_matrix(ds):
    return ds/np.nanmean(ds)

##################################################################
# functions for model ensemble subselection
##################################################################

# create csv with minimizing value and subset listed for each alpha-beta combo (one core)
def multi_run(m, cmip, im_or_em, season_region, alpha_steps, beta_steps, perf_cutoff,data):
    filename=Path(cmip+'_'+im_or_em+'_'+season_region+'_'+'alpha-beta-scan.csv')
    if filename.exists():
        raise RuntimeError('file exists!')
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha','beta','min_val']+[f'member{i}' for i in range(m)])
        for alpha_idx in range(0,alpha_steps+1):
            for beta_idx in range(0,beta_steps+1):
                alpha = alpha_idx/alpha_steps
                beta = beta_idx/beta_steps
                if alpha + beta > 1:
                    continue
                min_val, min_member = single_run(m, alpha, beta, perf_cutoff, data, silent=True)
                print(alpha, beta, min_val, min_member)
                writer.writerow([alpha,beta,min_val]+min_member)
    return filename
# finds minimizing subset
def single_run(m, alpha, beta, perf_cutoff, data, silent=False):
    n = len(data.delta_q)
    perf = xr.DataArray(data.delta_q.data, dims=['member'], coords=dict(member=data.delta_q.coords['member']))
    change_data = data.change.data
    dist_data = data.delta_i.data
    change = xr.DataArray(change_data, dims=['member','member_model'], coords=dict(member=data.change.coords['member'],member_model=data.change.coords['member_model']))
    dist = xr.DataArray(dist_data, dims=['member','member_model'], coords=dict(member=data.delta_i.coords['member'],member_model=data.delta_i.coords['member_model']))
    min_val, min_members = get_best_m_models(perf, dist, change, m, alpha, beta, perf_cutoff, silent=silent)
    return min_val, min_members

# normalizing metrics so they contribute equally to the cost function
def norm_matrices(perf, dist, change, perf_cutoff):
    members = list(perf.where(perf<perf_cutoff, drop=True).member.data) # drop members above the performance threshold
    n = len(members)
    perf = perf.sel(member=members)
    dist = dist.sel(member=members, member_model=members)
    change = change.sel(member=members, member_model=members)

    norm_dist = (dist - np.nanmean(dist))/np.nanstd(dist)/2 # /2 is due to double count ij, ji
    norm_change = (change - np.nanmean(change))/np.nanstd(change)/2

    # to save operations, we store the performance on the diagonal elements of the distance matrix
    # and pre-calculate the alpha mix.
    perf_diag = xr.DataArray(np.diag(perf.data),dims=['member','member_model'],coords=dict(member=members, member_model=members))
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            perf_diag[i,j]=np.nan

    norm_perf = (perf_diag - np.nanmean(perf_diag))/np.nanstd(perf_diag)

    for i in range(n):
        for j in range(n):
            if i==j:
                norm_dist[i,j] = 0
                norm_change[i,j] = 0
            else:
                norm_perf[i,j]=0

    return norm_perf, norm_dist, norm_change

# check all combinations to determine the cost-function-minimizing subset
def get_best_m_models(perf, dist, change, m, alpha, beta, perf_cutoff, silent=True):
    members = list(perf.where(perf<perf_cutoff, drop=True).member.data)
    n = len(members)
    if not silent:
        print(f'using {n} models with perf < {perf_cutoff}')

    norm_perf, norm_dist, norm_change = norm_matrices(perf, dist, change, perf_cutoff)
    cost_matrix = (1-alpha-beta) * norm_perf - alpha * norm_dist - beta * norm_change

    def cost_function(combo):
        return cost_matrix.isel(member=combo, member_model=combo).sum()

    # now we check for all combinations (n choose m) many
    min_val, min_combo = np.inf, []

    total_combinations = int(math.factorial(n) / math.factorial(m) / math.factorial(n-m))
    start_time = time.time()

    for i, combo in enumerate(itertools.combinations(range(len(members)), m)):
        cost = cost_function(list(combo))
        if cost < min_val:
            min_combo = combo
            min_val = cost.data
            min_members = [members[i] for i in min_combo]

    # this part displays progress, requires silent = False
        if not silent and i & 0b1111111111111 == 0:
            if i == 0:
                continue
            percent = i / total_combinations
            eta = (1-percent) * (time.time() - start_time) / percent
            print(f"{100*percent:>4.1f}% / eta in {eta/60:.1f} min / best score {min_val:.3f}")
            print(f"{', '.join(min_members)}")

    if not silent:
        print(f"all {total_combinations} combinations tested, which took {(time.time() - start_time)/60:.1f} min")
        print(f"min val (alpha={alpha}): {min_val}")
        print(f"min members:")
        for index, member in zip(min_combo, min_members):
            distances = [f"{dist[index, i].data:>6.2f}" for i in min_combo]
            spreads = [f"{change[index, i].data:>6.2f}" for i in min_combo]
            print(f" * {member:>24}   perf: {perf[index].data:>6.2f} dist: {' '.join(distances)} spread: {' '.join(spreads)}") # avr_dist: {avr_dist[index].data:>6.2f}
    return min_val, min_members

# creates csv in parallel (when multiple cores are available)
def multi_parallel_run(m, cmip, im_or_em, season_region, alpha_steps, beta_steps, perf_cutoff, data, max_workers):
    single_run_subdir = cmip+'_'+season_region+'_'+im_or_em
    print(f'running with {max_workers} workers.')
    filename=Path(cmip+'_'+im_or_em+'_'+season_region+'_'+'alpha-beta-scan.csv')
    if filename.exists():
        raise RuntimeError('file exists!')

    single_run_res = filename.parent / "single_run_res"
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for alpha_idx in range(0,alpha_steps+1):
            for beta_idx in range(0,beta_steps+1):
                alpha = alpha_idx/alpha_steps
                beta = beta_idx/beta_steps
                if alpha + beta > 1:
                    continue
                single_run_file = single_run_res / single_run_subdir / str(m) / str(alpha) / f'{beta}.csv'
                single_run_file.parent.mkdir(parents=True, exist_ok=True)
                if single_run_file.exists():
                    continue
                future = pool.submit(single_run_with_save, single_run_file, m, alpha, beta, perf_cutoff, data, silent=True)
                futures.append(future)
                print(f'submitted {alpha_idx}/{beta_idx}')

    for i, future in enumerate(futures):
        future.result()
        print('Progress', i, len(futures))

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha','beta','min_val']+[f'member{i}' for i in range(m)])
        for alpha_idx in range(0,alpha_steps+1):
            for beta_idx in range(0,beta_steps+1):
                alpha = alpha_idx/alpha_steps
                beta = beta_idx/beta_steps
                if alpha + beta > 1:
                    continue
                single_run_file = single_run_res / single_run_subdir / str(m) / str(alpha) / f'{beta}.csv'
                with open(single_run_file, 'r') as f2:
                    for row in csv.reader(f2):
                        print(row)
                        writer.writerow(row)
    return filename

# saves as an intermidiate step when running in parallel
def single_run_with_save(filename, m, alpha, beta, perf_cutoff, data, silent=False):
    min_val, min_member = single_run(m, alpha, beta, perf_cutoff, data, silent=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([alpha,beta,min_val]+min_member)

# ################################
# Make output files
# ################################

def make_output_file(dsDeltaQ,ds_spread_metric,targets,dsWi,outfile='perf_ind_spread_metrics.nc'):
    print('Models Used: ',dsWi.member.data)
    dsWi = dsWi.to_dataset(name='delta_i')
    dsWi['delta_q'] = dsDeltaQ
    dsWi['change'] = ds_spread_metric
    dsWi['tas_change'] = targets[0]
    dsWi['pr_change'] = targets[1]
    dsWi.to_netcdf(outfile)

def select_models(outfile, cmip, im_or_em, season_region, m, alpha_steps, beta_steps, perf_cutoff,max_workers=1):
    data = xr.open_dataset(outfile,use_cftime = True)
    if max_workers==1:
        return multi_run(m, cmip, im_or_em, season_region, alpha_steps, beta_steps, perf_cutoff,data)
    else:
        return multi_parallel_run(m, cmip, im_or_em, season_region, alpha_steps, beta_steps, perf_cutoff,data,max_workers)
