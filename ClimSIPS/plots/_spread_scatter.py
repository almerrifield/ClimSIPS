#################################
# packages
#################################

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from .. import member_selection as csms


__all__ = ["spread_scatter"]

def spread_scatter(outfile,cmip,im_or_em,season_region,spread_path,plotname="spread_scatter.png"):
    if cmip not in ['CMIP5','CMIP6','CH202x','CH202x_CMIP6','RCM']:
        raise NotImplementedError(cmip)
    if im_or_em not in ['IM','EM']:
        raise NotImplementedError(im_or_em)
    if season_region not in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH','JJA_ALPS','DJF_ALPS']:
        raise NotImplementedError(season_region)

    # ################################################
    fig = plt.figure(figsize=(9,7))
    ax = plt.subplot(111)
    # ################################################
    dsWi = xr.open_dataset(outfile,use_cftime = True)

    # All keyword arguments for CMIP5 and CMIP6 (add here)
    plot_kwargs = {
        'ACCESS-CM2-r1i1p1f1': dict(c='tab:red',s=20,marker='x',alpha=1), #
        'ACCESS-CM2-r2i1p1f1': dict(c='tab:red',s=20,marker='x',alpha=1), # start CMIP6 IM
        'ACCESS-CM2-r3i1p1f1': dict(c='tab:red',s=20,marker='x',alpha=1), # JJA CH202x case
        'ACCESS-ESM1-5-r1i1p1f1': dict(c='tab:red',s=30,marker='+',alpha=1),
        'ACCESS-ESM1-5-r5i1p1f1': dict(c='tab:red',s=30,marker='+',alpha=1), # DJF case
        'ACCESS-ESM1-5-r4i1p1f1': dict(c='tab:red',s=30,marker='+',alpha=1), # DJF CEU case
        'AWI-CM-1-1-MR-r1i1p1f1': dict(c='tab:orange',s=20,marker='x',alpha=1),
        'CAS-ESM2-0-r1i1p1f1': dict(c='tab:cyan',s=20,marker='x',alpha=1),
        'CAS-ESM2-0-r3i1p1f1': dict(c='tab:cyan',s=20,marker='x',alpha=1), # DJF case
        'CESM2-WACCM-r1i1p1f1': dict(c='darkgoldenrod',s=20,marker='x',alpha=1),
        'CESM2-WACCM-r2i1p1f1': dict(c='darkgoldenrod',s=20,marker='x',alpha=1),
        'CESM2-WACCM-r3i1p1f1': dict(c='darkgoldenrod',s=20,marker='x',alpha=1), # DJF CEU case
        'CESM2-r11i1p1f1': dict(c='darkgoldenrod',s=20,marker='o',alpha=1),
        'CESM2-r1i1p1f1': dict(c='darkgoldenrod',s=20,marker='o',alpha=1), # JJA CH202x case
        'CESM2-r2i1p1f1': dict(c='darkgoldenrod',s=20,marker='o',alpha=1), # DJF case
        'CMCC-CM2-SR5-r1i1p1f1': dict(c='darkgoldenrod',s=20,marker='^',alpha=1),
        'CMCC-ESM2-r1i1p1f1': dict(c='darkgoldenrod',s=30,marker='*',alpha=1),
        'CNRM-CM6-1-HR-r1i1p1f2': dict(c='cornflowerblue',s=20,marker='x',alpha=1),
        'CNRM-CM6-1-r1i1p1f2': dict(c='cornflowerblue',s=20,marker='o',alpha=1), # JJA CH202x case
        'CNRM-CM6-1-r4i1p1f2': dict(c='cornflowerblue',s=20,marker='o',alpha=1),
        'CNRM-CM6-1-r5i1p1f2': dict(c='cornflowerblue',s=20,marker='o',alpha=1), # DJF case
        'CNRM-CM6-1-r6i1p1f2': dict(c='cornflowerblue',s=20,marker='o',alpha=1),
        'CNRM-ESM2-1-r2i1p1f2': dict(c='cornflowerblue',s=20,marker='^',alpha=1),
        'CNRM-ESM2-1-r3i1p1f2': dict(c='cornflowerblue',s=20,marker='^',alpha=1), # DJF case
        'CNRM-ESM2-1-r4i1p1f2': dict(c='cornflowerblue',s=20,marker='^',alpha=1),
        'CanESM5-r16i1p1f1': dict(c='dodgerblue',s=20,marker='x',alpha=1),
        'CanESM5-r10i1p2f1': dict(c='dodgerblue',s=20,marker='x',alpha=1), # JJA CH202x case
        'CanESM5-r14i1p2f1': dict(c='dodgerblue',s=20,marker='x',alpha=1), # DJF case
        'CanESM5-r1i1p2f1': dict(c='dodgerblue',s=20,marker='x',alpha=1), # DJF CEU case
        'CanESM5-r23i1p2f1': dict(c='dodgerblue',s=20,marker='x',alpha=1),
        'E3SM-1-1-r1i1p1f1': dict(c='k',s=20,marker='x',alpha=1),
        'FGOALS-f3-L-r1i1p1f1': dict(c='maroon',s=20,marker='x',alpha=1),
        'FGOALS-g3-r1i1p1f1': dict(c='maroon',s=20,marker='o',alpha=1), # JJA CH202x case
        'FGOALS-g3-r2i1p1f1': dict(c='maroon',s=20,marker='o',alpha=1),
        'GFDL-CM4-r1i1p1f1': dict(c='indigo',s=20,marker='x',alpha=1),
        'GFDL-ESM4-r1i1p1f1': dict(c='indigo',s=20,marker='o',alpha=1),
        'GISS-E2-1-G-r1i1p3f1': dict(c='blueviolet',s=20,marker='x',alpha=1),
        'HadGEM3-GC31-LL-r3i1p1f3': dict(c='tab:red',s=20,marker='o',alpha=1),
        'HadGEM3-GC31-LL-r4i1p1f3': dict(c='tab:red',s=20,marker='o',alpha=1), # JJA CH202x case
        'HadGEM3-GC31-MM-r1i1p1f3': dict(c='tab:red',s=20,marker='^',alpha=1),
        'HadGEM3-GC31-MM-r2i1p1f3': dict(c='tab:red',s=20,marker='^',alpha=1), # DJF case
        'INM-CM4-8-r1i1p1f1': dict(c='mediumseagreen',s=20,marker='x',alpha=1),
        'INM-CM5-0-r1i1p1f1': dict(c='mediumseagreen',s=20,marker='o',alpha=1),
        'IPSL-CM6A-LR-r6i1p1f1': dict(c='royalblue',s=20,marker='x',alpha=1),
        'IPSL-CM6A-LR-r2i1p1f1': dict(c='royalblue',s=20,marker='x',alpha=1), # DJF case
        'IPSL-CM6A-LR-r3i1p1f1': dict(c='royalblue',s=20,marker='x',alpha=1), # JJA CH202x case
        'IPSL-CM6A-LR-r4i1p1f1': dict(c='royalblue',s=20,marker='x',alpha=1), # DJF CEU case
        'KACE-1-0-G-r3i1p1f1': dict(c='tab:red',s=30,marker='*',alpha=1),
        'KIOST-ESM-r1i1p1f1': dict(c='darkslateblue',s=20,marker='x',alpha=1),
        'MIROC-ES2L-r1i1p1f2': dict(c='lightsalmon',s=20,marker='x',alpha=1),
        'MIROC-ES2L-r9i1p1f2': dict(c='lightsalmon',s=20,marker='x',alpha=1), # DJF case
        'MIROC-ES2L-r2i1p1f2': dict(c='lightsalmon',s=20,marker='x',alpha=1), # DJF CEU case
        'MIROC6-r15i1p1f1': dict(c='lightsalmon',s=20,marker='o',alpha=1),
        'MIROC6-r12i1p1f1': dict(c='lightsalmon',s=20,marker='o',alpha=1), # DJF case
        'MIROC6-r14i1p1f1': dict(c='lightsalmon',s=20,marker='o',alpha=1),
        'MIROC6-r26i1p1f1': dict(c='lightsalmon',s=20,marker='o',alpha=1), # JJA CH202x case
        'MIROC6-r50i1p1f1': dict(c='lightsalmon',s=20,marker='o',alpha=1), # DJF CEU case
        'MPI-ESM1-2-HR-r1i1p1f1': dict(c='tab:orange',s=20,marker='o',alpha=1),
        'MPI-ESM1-2-HR-r2i1p1f1': dict(c='tab:orange',s=20,marker='o',alpha=1), # DJF case
        'MPI-ESM1-2-LR-r2i1p1f1': dict(c='tab:orange',s=20,marker='^',alpha=1), # JJA CH202x case
        'MPI-ESM1-2-LR-r10i1p1f1': dict(c='tab:orange',s=20,marker='^',alpha=1),
        'MPI-ESM1-2-LR-r4i1p1f1': dict(c='tab:orange',s=20,marker='^',alpha=1), # DJF case
        'MPI-ESM1-2-LR-r5i1p1f1': dict(c='tab:orange',s=20,marker='^',alpha=1),
        'MPI-ESM1-2-LR-r9i1p1f1': dict(c='tab:orange',s=20,marker='^',alpha=1), # DJF CEU case
        'MRI-ESM2-0-r1i1p1f1': dict(c='palevioletred',s=20,marker='x',alpha=1),
        'MRI-ESM2-0-r1i2p1f1': dict(c='palevioletred',s=20,marker='x',alpha=1),
        'NESM3-r1i1p1f1': dict(c='tab:orange',s=30,marker='*',alpha=1),
        'NESM3-r2i1p1f1': dict(c='tab:orange',s=30,marker='*',alpha=1),
        'NorESM2-MM-r1i1p1f1': dict(c='darkgoldenrod',s=20,marker='d',alpha=1),
        'TaiESM1-r1i1p1f1': dict(c='darkgoldenrod',s=30,marker='+',alpha=1),
        'UKESM1-0-LL-r1i1p1f2': dict(c='tab:red',s=20,marker='d',alpha=1), # end CMIP6 IM
        ##############################
        'UKESM1-0-LL-r2i1p1f2': dict(c='tab:red',s=20,marker='d',alpha=1), # DJF case
        'UKESM1-0-LL-r3i1p1f2': dict(c='tab:red',s=20,marker='d',alpha=1), # DJF CEU case
        'ACCESS-CM2-r0i0p0f0': dict(c='tab:red',s=20,marker='x',alpha=1), # start CMIP6 EM
        'ACCESS-ESM1-5-r0i0p0f0': dict(c='tab:red',s=30,marker='+',alpha=1),
        'AWI-CM-1-1-MR-r1i1p1f1': dict(c='tab:orange',s=20,marker='x',alpha=1),
        'CAS-ESM2-0-r0i0p0f0': dict(c='tab:cyan',s=20,marker='x',alpha=1),
        'CESM2-WACCM-r0i0p0f0': dict(c='darkgoldenrod',s=20,marker='x',alpha=1),
        'CESM2-r0i0p0f0': dict(c='darkgoldenrod',s=20,marker='o',alpha=1),
        'CMCC-CM2-SR5-r1i1p1f1': dict(c='darkgoldenrod',s=20,marker='^',alpha=1),
        'CMCC-ESM2-r1i1p1f1': dict(c='darkgoldenrod',s=30,marker='*',alpha=1),
        'CNRM-CM6-1-HR-r1i1p1f2': dict(c='cornflowerblue',s=20,marker='x',alpha=1),
        'CNRM-CM6-1-r0i0p0f0': dict(c='cornflowerblue',s=20,marker='o',alpha=1),
        'CNRM-ESM2-1-r0i0p0f0': dict(c='cornflowerblue',s=20,marker='^',alpha=1),
        'CanESM5-r0i0p0f0': dict(c='dodgerblue',s=20,marker='x',alpha=1),
        'E3SM-1-1-r1i1p1f1': dict(c='k',s=20,marker='x',alpha=1),
        'FGOALS-f3-L-r1i1p1f1': dict(c='maroon',s=20,marker='x',alpha=1),
        'FGOALS-g3-r0i0p0f0': dict(c='maroon',s=20,marker='o',alpha=1),
        'GFDL-CM4-r1i1p1f1': dict(c='indigo',s=20,marker='x',alpha=1),
        'GFDL-ESM4-r1i1p1f1': dict(c='indigo',s=20,marker='o',alpha=1),
        'GISS-E2-1-G-r1i1p3f1': dict(c='blueviolet',s=20,marker='x',alpha=1),
        'HadGEM3-GC31-LL-r0i0p0f0': dict(c='tab:red',s=20,marker='o',alpha=1),
        'HadGEM3-GC31-MM-r0i0p0f0': dict(c='tab:red',s=20,marker='^',alpha=1),
        'INM-CM4-8-r1i1p1f1': dict(c='mediumseagreen',s=20,marker='x',alpha=1),
        'INM-CM5-0-r1i1p1f1': dict(c='mediumseagreen',s=20,marker='o',alpha=1),
        'IPSL-CM6A-LR-r0i0p0f0': dict(c='royalblue',s=20,marker='x',alpha=1),
        'KACE-1-0-G-r0i0p0f0': dict(c='tab:red',s=30,marker='*',alpha=1),
        'KIOST-ESM-r1i1p1f1': dict(c='darkslateblue',s=20,marker='x',alpha=1),
        'MIROC-ES2L-r0i0p0f0': dict(c='lightsalmon',s=20,marker='x',alpha=1),
        'MIROC6-r0i0p0f0': dict(c='lightsalmon',s=20,marker='o',alpha=1),
        'MPI-ESM1-2-HR-r0i0p0f0': dict(c='tab:orange',s=20,marker='o',alpha=1),
        'MPI-ESM1-2-LR-r0i0p0f0': dict(c='tab:orange',s=20,marker='^',alpha=1),
        'MRI-ESM2-0-r0i0p0f0': dict(c='palevioletred',s=20,marker='x',alpha=1),
        'NESM3-r0i0p0f0': dict(c='tab:orange',s=30,marker='*',alpha=1),
        'NorESM2-MM-r1i1p1f1': dict(c='darkgoldenrod',s=20,marker='d',alpha=1),
        'TaiESM1-r1i1p1f1': dict(c='darkgoldenrod',s=30,marker='+',alpha=1),
        'UKESM1-0-LL-r0i0p0f0': dict(c='tab:red',s=20,marker='d',alpha=1), # end CMIP6 EM
        ##############################
        'ACCESS1-0-r1i1p1': dict(c='tab:red',s=20,marker='x',alpha=1), # start CMIP5 IM
        'ACCESS1-3-r1i1p1': dict(c='tab:red',s=20,marker='o',alpha=1),
        'CCSM4-r6i1p1': dict(c='darkgoldenrod',s=20,marker='x',alpha=1),
        'CCSM4-r5i1p1': dict(c='darkgoldenrod',s=20,marker='x',alpha=1), # DJF case
        'CCSM4-r2i1p1': dict(c='darkgoldenrod',s=20,marker='x',alpha=1), # JJA_CH case
        'CESM1-CAM5-r1i1p1': dict(c='darkgoldenrod',s=20,marker='o',alpha=1),
        'CESM1-CAM5-r3i1p1': dict(c='darkgoldenrod',s=20,marker='o',alpha=1), # DJF case
        'CNRM-CM5-r1i1p1': dict(c='cornflowerblue',s=20,marker='x',alpha=1), # RCM case
        'CNRM-CM5-r2i1p1': dict(c='cornflowerblue',s=20,marker='x',alpha=1),
        'CNRM-CM5-r4i1p1': dict(c='cornflowerblue',s=20,marker='x',alpha=1), # DJF case
        'CSIRO-Mk3-6-0-r10i1p1': dict(c='deeppink',s=20,marker='x',alpha=1),
        'CSIRO-Mk3-6-0-r1i1p1': dict(c='deeppink',s=20,marker='x',alpha=1), # DJF CH
        'CanESM2-r5i1p1': dict(c='dodgerblue',s=20,marker='x',alpha=1),
        'CanESM2-r3i1p1': dict(c='dodgerblue',s=20,marker='x',alpha=1), # DJF case
        'CanESM2-r1i1p1': dict(c='dodgerblue',s=20,marker='x',alpha=1), # RCM case
        'CanESM2-r2i1p1': dict(c='dodgerblue',s=20,marker='x',alpha=1), # JJA_CH case
        'CanESM2-r4i1p1': dict(c='dodgerblue',s=20,marker='x',alpha=1), # DJF_CH case
        'GFDL-CM3-r1i1p1': dict(c='indigo',s=20,marker='x',alpha=1),
        'GFDL-ESM2G-r1i1p1': dict(c='indigo',s=20,marker='o',alpha=1),
        'GFDL-ESM2M-r1i1p1': dict(c='indigo',s=20,marker='^',alpha=1),
        'GISS-E2-H-r1i1p3': dict(c='blueviolet',s=20,marker='x',alpha=1),
        'GISS-E2-H-r1i1p2': dict(c='blueviolet',s=20,marker='x',alpha=1), #JJA_CH
        'GISS-E2-H-r2i1p1': dict(c='blueviolet',s=20,marker='x',alpha=1), #DJF_CH
        'GISS-E2-R-r1i1p1': dict(c='blueviolet',s=20,marker='o',alpha=1), #JJA_CH
        'GISS-E2-R-r1i1p3': dict(c='blueviolet',s=20,marker='o',alpha=1),
        'GISS-E2-R-r2i1p3': dict(c='blueviolet',s=20,marker='o',alpha=1), # DJF case
        'HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='^',alpha=1), # RCM case
        'HadGEM2-ES-r4i1p1': dict(c='tab:red',s=20,marker='^',alpha=1),
        'HadGEM2-ES-r2i1p1': dict(c='tab:red',s=20,marker='^',alpha=1), # DJF_CH
        'IPSL-CM5A-LR-r1i1p1': dict(c='royalblue',s=20,marker='x',alpha=1),
        'IPSL-CM5A-LR-r2i1p1': dict(c='royalblue',s=20,marker='x',alpha=1), ## alternative, similar spread generator
        'IPSL-CM5A-LR-r3i1p1': dict(c='royalblue',s=20,marker='x',alpha=1), # DJF case
        'IPSL-CM5A-LR-r4i1p1': dict(c='royalblue',s=20,marker='x',alpha=1), # JJA_CH case
        'IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='o',alpha=1),
        'IPSL-CM5B-LR-r1i1p1': dict(c='royalblue',s=20,marker='^',alpha=1),
        'MIROC-ESM-r1i1p1': dict(c='lightsalmon',s=20,marker='x',alpha=1),
        'MIROC5-r1i1p1': dict(c='lightsalmon',s=20,marker='o',alpha=1), # RCM case
        'MIROC5-r2i1p1': dict(c='lightsalmon',s=20,marker='o',alpha=1),# JJA_CH
        'MIROC5-r3i1p1': dict(c='lightsalmon',s=20,marker='o',alpha=1),
        'MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='x',alpha=1), # DJF case
        'MPI-ESM-LR-r2i1p1': dict(c='tab:orange',s=20,marker='o',alpha=1),
        'MPI-ESM-LR-r3i1p1': dict(c='tab:orange',s=20,marker='^',alpha=1),
        'MPI-ESM-MR-r1i1p1': dict(c='tab:orange',s=20,marker='o',alpha=1),
        'MRI-CGCM3-r1i1p1': dict(c='palevioletred',s=20,marker='x',alpha=1),
        'NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='^',alpha=1),
        'NorESM1-ME-r1i1p1': dict(c='darkgoldenrod',s=30,marker='*',alpha=1),
        'bcc-csm1-1-m-r1i1p1': dict(c='silver',s=20,marker='x',alpha=1),
        'bcc-csm1-1-r1i1p1': dict(c='silver',s=20,marker='o',alpha=1),
        'inmcm4-r1i1p1': dict(c='mediumseagreen',s=20,marker='x',alpha=1),
        'EC-EARTH-r1i1p1': dict(c='darkgreen',s=20,marker='x',alpha=1), # end CMIP5 IM
        'EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='o',alpha=1), # JJA_CH
        'EC-EARTH-r0i0p0': dict(c='darkgreen',s=20,marker='x',alpha=1),
        ##############################
        'ACCESS1-0-r1i1p1': dict(c='tab:red',s=20,marker='x',alpha=1), # start CMIP5 EM
        'ACCESS1-3-r1i1p1': dict(c='tab:red',s=20,marker='o',alpha=1),
        'CCSM4-r0i0p0': dict(c='darkgoldenrod',s=20,marker='x',alpha=1),
        'CESM1-CAM5-r0i0p0': dict(c='darkgoldenrod',s=20,marker='o',alpha=1),
        'CNRM-CM5-r0i0p0': dict(c='cornflowerblue',s=20,marker='x',alpha=1),
        'CSIRO-Mk3-6-0-r0i0p0': dict(c='deeppink',s=20,marker='x',alpha=1),
        'CanESM2-r0i0p0': dict(c='dodgerblue',s=20,marker='x',alpha=1),
        'GFDL-CM3-r1i1p1': dict(c='indigo',s=20,marker='x',alpha=1),
        'GFDL-ESM2G-r1i1p1': dict(c='indigo',s=20,marker='o',alpha=1),
        'GFDL-ESM2M-r1i1p1': dict(c='indigo',s=20,marker='^',alpha=1),
        'GISS-E2-H-r0i0p0': dict(c='blueviolet',s=20,marker='x',alpha=1),
        'GISS-E2-R-r0i0p0': dict(c='blueviolet',s=20,marker='o',alpha=1),
        'HadGEM2-ES-r0i0p0': dict(c='tab:red',s=20,marker='^',alpha=1),
        'IPSL-CM5A-LR-r0i0p0': dict(c='royalblue',s=20,marker='x',alpha=1),
        'IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='o',alpha=1),
        'IPSL-CM5B-LR-r1i1p1': dict(c='royalblue',s=20,marker='^',alpha=1),
        'MIROC-ESM-r1i1p1': dict(c='lightsalmon',s=20,marker='x',alpha=1),
        'MIROC5-r0i0p0': dict(c='lightsalmon',s=20,marker='o',alpha=1),
        'MPI-ESM-LR-r0i0p0': dict(c='tab:orange',s=20,marker='x',alpha=1),
        'MPI-ESM-MR-r1i1p1': dict(c='tab:orange',s=20,marker='o',alpha=1),
        'MRI-CGCM3-r1i1p1': dict(c='palevioletred',s=20,marker='x',alpha=1),
        'NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='^',alpha=1),
        'NorESM1-ME-r1i1p1': dict(c='darkgoldenrod',s=30,marker='*',alpha=1),
        'bcc-csm1-1-m-r1i1p1': dict(c='silver',s=20,marker='x',alpha=1),
        'bcc-csm1-1-r1i1p1': dict(c='silver',s=20,marker='o',alpha=1),
        'inmcm4-r1i1p1': dict(c='mediumseagreen',s=20,marker='x',alpha=1), # end CMIP5 EM
        ##############################
        'CLMcom-CCLM4-8-17-CanESM2-r1i1p1': dict(c='dodgerblue',s=20,marker='x',alpha=1),
        'CLMcom-CCLM4-8-17-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='x',alpha=1),
        'CLMcom-CCLM4-8-17-EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='x',alpha=1),
        'CLMcom-CCLM4-8-17-MIROC5-r1i1p1': dict(c='tab:purple',s=20,marker='x',alpha=1),
        'CLMcom-CCLM4-8-17-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='x',alpha=1),
        'CLMcom-CCLM4-8-17-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='x',alpha=1),
        ##############################
        'CLMcom-ETH-COSMO-crCLIM-v1-1-CNRM-CM5-r1i1p1': dict(c='m',s=30,marker='+',alpha=1),
        'CLMcom-ETH-COSMO-crCLIM-v1-1-EC-EARTH-r12i1p1': dict(c='darkgreen',s=30,marker='+',alpha=1),
        'CLMcom-ETH-COSMO-crCLIM-v1-1-EC-EARTH-r1i1p1': dict(c='darkgreen',s=30,marker='+',alpha=1),
        'CLMcom-ETH-COSMO-crCLIM-v1-1-EC-EARTH-r3i1p1': dict(c='darkgreen',s=30,marker='+',alpha=1),
        'CLMcom-ETH-COSMO-crCLIM-v1-1-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=30,marker='+',alpha=1),
        'CLMcom-ETH-COSMO-crCLIM-v1-1-MPI-ESM-LR-r2i1p1': dict(c='tab:orange',s=30,marker='+',alpha=1),
        'CLMcom-ETH-COSMO-crCLIM-v1-1-MPI-ESM-LR-r3i1p1': dict(c='tab:orange',s=30,marker='+',alpha=1),
        'CLMcom-ETH-COSMO-crCLIM-v1-1-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=30,marker='+',alpha=1),
        ##############################
        'CNRM-ALADIN63-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='o',alpha=1),
        'CNRM-ALADIN63-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='o',alpha=1),
        'CNRM-ALADIN63-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='o',alpha=1),
        'CNRM-ALADIN63-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='o',alpha=1),
        ##############################
        'DMI-HIRHAM5-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='^',alpha=1),
        'DMI-HIRHAM5-EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='^',alpha=1),
        'DMI-HIRHAM5-EC-EARTH-r1i1p1': dict(c='darkgreen',s=20,marker='^',alpha=1),
        'DMI-HIRHAM5-EC-EARTH-r3i1p1': dict(c='darkgreen',s=20,marker='^',alpha=1),
        'DMI-HIRHAM5-IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='^',alpha=1),
        'DMI-HIRHAM5-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='^',alpha=1),
        'DMI-HIRHAM5-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='^',alpha=1),
        'DMI-HIRHAM5-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='^',alpha=1),
        ##############################
        'GERICS-REMO2015-CanESM2-r1i1p1': dict(c='dodgerblue',s=20,marker='*',alpha=1),
        'GERICS-REMO2015-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='*',alpha=1),
        'GERICS-REMO2015-EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='*',alpha=1),
        'GERICS-REMO2015-IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='*',alpha=1),
        'GERICS-REMO2015-MIROC5-r1i1p1': dict(c='tab:purple',s=20,marker='*',alpha=1),
        'GERICS-REMO2015-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='*',alpha=1),
        'GERICS-REMO2015-MPI-ESM-LR-r3i1p1': dict(c='tab:orange',s=20,marker='*',alpha=1),
        'GERICS-REMO2015-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='*',alpha=1),
        ##############################
        'ICTP-RegCM4-6-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='s',alpha=1), ## new
        'ICTP-RegCM4-6-EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='s',alpha=1),
        'ICTP-RegCM4-6-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='s',alpha=1),  ## new
        'ICTP-RegCM4-6-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='s',alpha=1),  ## new
        'ICTP-RegCM4-6-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='s',alpha=1),
        ##############################
        'IPSL-INERIS-WRF331F-IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='d',alpha=1),
        ##############################
        'IPSL-WRF381P-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='p',alpha=1),
        'IPSL-WRF381P-EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='p',alpha=1),
        'IPSL-WRF381P-IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='p',alpha=1),
        'IPSL-WRF381P-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='p',alpha=1), ## old
        'IPSL-WRF381P-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='p',alpha=1),
        'IPSL-WRF381P-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='p',alpha=1),
        ##############################
        'KNMI-RACMO22E-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='v',alpha=1),
        'KNMI-RACMO22E-EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='v',alpha=1),
        'KNMI-RACMO22E-EC-EARTH-r1i1p1': dict(c='darkgreen',s=20,marker='v',alpha=1),
        'KNMI-RACMO22E-EC-EARTH-r3i1p1': dict(c='darkgreen',s=20,marker='v',alpha=1),
        'KNMI-RACMO22E-IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='v',alpha=1),
        'KNMI-RACMO22E-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='v',alpha=1),
        'KNMI-RACMO22E-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='v',alpha=1),
        'KNMI-RACMO22E-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='v',alpha=1),
        ##############################
        'MOHC-HadREM3-GA7-05-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='D',alpha=1),
        'MOHC-HadREM3-GA7-05-EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='D',alpha=1),
        'MOHC-HadREM3-GA7-05-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='D',alpha=1),
        'MOHC-HadREM3-GA7-05-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='D',alpha=1),
        'MOHC-HadREM3-GA7-05-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='D',alpha=1),
        ##############################
        'MPI-CSC-REMO2009-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='_',alpha=1),
        'MPI-CSC-REMO2009-MPI-ESM-LR-r2i1p1': dict(c='tab:orange',s=20,marker='_',alpha=1),
        ##############################
        'RMIB-UGent-ALARO-0-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='|',alpha=1),
        ##############################
        'SMHI-RCA4-CNRM-CM5-r1i1p1': dict(c='m',s=20,marker='1',alpha=1),
        'SMHI-RCA4-EC-EARTH-r12i1p1': dict(c='darkgreen',s=20,marker='1',alpha=1),
        'SMHI-RCA4-EC-EARTH-r1i1p1': dict(c='darkgreen',s=20,marker='1',alpha=1),
        'SMHI-RCA4-EC-EARTH-r3i1p1': dict(c='darkgreen',s=20,marker='1',alpha=1),
        'SMHI-RCA4-IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='1',alpha=1),
        'SMHI-RCA4-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='1',alpha=1),
        'SMHI-RCA4-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='1',alpha=1),
        'SMHI-RCA4-MPI-ESM-LR-r2i1p1': dict(c='tab:orange',s=20,marker='1',alpha=1),
        'SMHI-RCA4-MPI-ESM-LR-r3i1p1': dict(c='tab:orange',s=20,marker='1',alpha=1),
        'SMHI-RCA4-NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='1',alpha=1),
        ##############################
        'UHOH-WRF361H-EC-EARTH-r1i1p1': dict(c='darkgreen',s=20,marker='>',alpha=1),
        'UHOH-WRF361H-MIROC5-r1i1p1': dict(c='tab:purple',s=20,marker='>',alpha=1),
        'UHOH-WRF361H-HadGEM2-ES-r1i1p1': dict(c='tab:red',s=20,marker='>',alpha=1),
        'UHOH-WRF361H-MPI-ESM-LR-r1i1p1': dict(c='tab:orange',s=20,marker='>',alpha=1),
        ##############################
        'CLMcom-ETH-COSMO-crCLIM-v1-1-EC-EARTH-r0i0p0': dict(c='darkgreen',s=30,marker='+',alpha=1),
        'CLMcom-ETH-COSMO-crCLIM-v1-1-MPI-ESM-LR-r0i0p0': dict(c='tab:orange',s=30,marker='+',alpha=1),
        'DMI-HIRHAM5-EC-EARTH-r0i0p0': dict(c='darkgreen',s=20,marker='^',alpha=1),
        'IPSL-INERIS-WRF331F-IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='d',alpha=1),
        'KNMI-RACMO22E-EC-EARTH-r0i0p0': dict(c='darkgreen',s=20,marker='v',alpha=1),
        'MPI-CSC-REMO2009-MPI-ESM-LR-r0i0p0': dict(c='tab:orange',s=20,marker='_',alpha=1),
        'SMHI-RCA4-EC-EARTH-r0i0p0': dict(c='darkgreen',s=20,marker='1',alpha=1),
        'SMHI-RCA4-MPI-ESM-LR-r0i0p0': dict(c='tab:orange',s=20,marker='1',alpha=1),
        }


    if cmip == 'CMIP6' and im_or_em == 'IM' and season_region in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH']:
        models =  csms.CMIP6_spread_maximizing_members(csms.CMIP6_common_members,season_region,spread_path)
        period = '2041/2060 - 1995/2014'
    if cmip == 'CMIP6' and im_or_em == 'EM' and season_region in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH']:
        models = ['ACCESS-CM2-r0i0p0f0', 'ACCESS-ESM1-5-r0i0p0f0',
            'AWI-CM-1-1-MR-r1i1p1f1', 'CAS-ESM2-0-r0i0p0f0', 'CESM2-WACCM-r0i0p0f0',
            'CESM2-r0i0p0f0', 'CMCC-CM2-SR5-r1i1p1f1', 'CMCC-ESM2-r1i1p1f1',
            'CNRM-CM6-1-HR-r1i1p1f2', 'CNRM-CM6-1-r0i0p0f0', 'CNRM-ESM2-1-r0i0p0f0',
            'CanESM5-r0i0p0f0', 'E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1',
            'FGOALS-g3-r0i0p0f0', 'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1',
            'GISS-E2-1-G-r1i1p3f1', 'HadGEM3-GC31-LL-r0i0p0f0',
            'HadGEM3-GC31-MM-r0i0p0f0', 'INM-CM4-8-r1i1p1f1', 'INM-CM5-0-r1i1p1f1',
            'IPSL-CM6A-LR-r0i0p0f0', 'KACE-1-0-G-r0i0p0f0', 'KIOST-ESM-r1i1p1f1',
            'MIROC-ES2L-r0i0p0f0', 'MIROC6-r0i0p0f0', 'MPI-ESM1-2-HR-r0i0p0f0',
            'MPI-ESM1-2-LR-r0i0p0f0', 'MRI-ESM2-0-r0i0p0f0', 'NESM3-r0i0p0f0',
            'NorESM2-MM-r1i1p1f1', 'TaiESM1-r1i1p1f1', 'UKESM1-0-LL-r0i0p0f0']
        period = '2041/2060 - 1995/2014'
    if cmip == 'CMIP5' and im_or_em == 'IM' and season_region in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH']:
        models =  csms.CMIP5_spread_maximizing_members(csms.CMIP5_common_members,season_region,spread_path)
        period = '2041/2060 - 1995/2014'
    if cmip == 'CMIP5' and im_or_em == 'EM' and season_region in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH']:
        models = ['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'CCSM4-r0i0p0',
            'CESM1-CAM5-r0i0p0', 'CNRM-CM5-r0i0p0', 'CSIRO-Mk3-6-0-r0i0p0',
            'CanESM2-r0i0p0', 'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1',
            'GFDL-ESM2M-r1i1p1', 'GISS-E2-H-r0i0p0', 'GISS-E2-R-r0i0p0',
            'HadGEM2-ES-r0i0p0', 'IPSL-CM5A-LR-r0i0p0', 'IPSL-CM5A-MR-r1i1p1',
            'IPSL-CM5B-LR-r1i1p1', 'MIROC-ESM-r1i1p1', 'MIROC5-r0i0p0',
            'MPI-ESM-LR-r0i0p0', 'MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1',
            'NorESM1-M-r1i1p1', 'NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1',
            'bcc-csm1-1-r1i1p1', 'inmcm4-r1i1p1']
        period = '2041/2060 - 1995/2014'
    if cmip == 'CH202x' and im_or_em == 'IM' and season_region in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH']:
        models =  csms.CMIP5_RCM_spread_maximizing_members(csms.CMIP5_RCM_common_members,season_region,spread_path)
        period = '2070/2099 - 1981/2010'
    if cmip == 'CH202x' and im_or_em == 'EM' and season_region in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH']:
        models = ['CNRM-CM5-r1i1p1','CanESM2-r1i1p1','EC-EARTH-r0i0p0','HadGEM2-ES-r1i1p1','IPSL-CM5A-MR-r1i1p1','MIROC5-r1i1p1',
        'MPI-ESM-LR-r0i0p0','NorESM1-M-r1i1p1']
        period = '2070/2099 - 1981/2010'
    if cmip == 'CH202x_CMIP6' and im_or_em == 'IM' and season_region in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH']:
        models =  csms.CMIP6_max_warming_members(csms.CMIP6_common_members,season_region,spread_path)
        period = '2070/2099 - 1981/2010'
    if cmip == 'CH202x_CMIP6' and im_or_em == 'EM' and season_region in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH']:
        models = ['ACCESS-CM2-r0i0p0f0', 'ACCESS-ESM1-5-r0i0p0f0',
            'AWI-CM-1-1-MR-r1i1p1f1', 'CAS-ESM2-0-r0i0p0f0', 'CESM2-WACCM-r0i0p0f0',
            'CESM2-r0i0p0f0', 'CMCC-CM2-SR5-r1i1p1f1', 'CMCC-ESM2-r1i1p1f1',
            'CNRM-CM6-1-HR-r1i1p1f2', 'CNRM-CM6-1-r0i0p0f0', 'CNRM-ESM2-1-r0i0p0f0',
            'CanESM5-r0i0p0f0', 'E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1',
            'FGOALS-g3-r0i0p0f0', 'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1',
            'GISS-E2-1-G-r1i1p3f1', 'HadGEM3-GC31-LL-r0i0p0f0',
            'HadGEM3-GC31-MM-r0i0p0f0', 'INM-CM4-8-r1i1p1f1', 'INM-CM5-0-r1i1p1f1',
            'IPSL-CM6A-LR-r0i0p0f0', 'KACE-1-0-G-r0i0p0f0', 'KIOST-ESM-r1i1p1f1',
            'MIROC-ES2L-r0i0p0f0', 'MIROC6-r0i0p0f0', 'MPI-ESM1-2-HR-r0i0p0f0',
            'MPI-ESM1-2-LR-r0i0p0f0', 'MRI-ESM2-0-r0i0p0f0', 'NESM3-r0i0p0f0',
            'NorESM2-MM-r1i1p1f1', 'TaiESM1-r1i1p1f1', 'UKESM1-0-LL-r0i0p0f0']
        period = '2070/2099 - 1981/2010'
    if cmip == 'RCM' and im_or_em == 'IM' and season_region in ['JJA_ALPS','DJF_ALPS','JJA_CH','DJF_CH']:
        models =  csms.RCM_max_warming_members(csms.RCM_common_members,season_region,spread_path)
        period = '2070/2099 - 1981/2010'
    if cmip == 'RCM' and im_or_em == 'EM' and season_region in ['JJA_ALPS','DJF_ALPS','JJA_CH','DJF_CH']:
        models = ['CLMcom-CCLM4-8-17-CanESM2-r1i1p1', 'CLMcom-CCLM4-8-17-EC-EARTH-r12i1p1', 'CLMcom-CCLM4-8-17-HadGEM2-ES-r1i1p1', 'CLMcom-CCLM4-8-17-MIROC5-r1i1p1',
                     'CLMcom-CCLM4-8-17-MPI-ESM-LR-r1i1p1', 'CLMcom-ETH-COSMO-crCLIM-v1-1-CNRM-CM5-r1i1p1', 'CLMcom-ETH-COSMO-crCLIM-v1-1-EC-EARTH-r0i0p0', 'CLMcom-ETH-COSMO-crCLIM-v1-1-MPI-ESM-LR-r0i0p0',
                     'CLMcom-ETH-COSMO-crCLIM-v1-1-NorESM1-M-r1i1p1', 'CNRM-ALADIN63-CNRM-CM5-r1i1p1', 'CNRM-ALADIN63-HadGEM2-ES-r1i1p1', 'CNRM-ALADIN63-MPI-ESM-LR-r1i1p1', 'CNRM-ALADIN63-NorESM1-M-r1i1p1',
                     'DMI-HIRHAM5-CNRM-CM5-r1i1p1', 'DMI-HIRHAM5-EC-EARTH-r0i0p0', 'DMI-HIRHAM5-HadGEM2-ES-r1i1p1', 'DMI-HIRHAM5-IPSL-CM5A-MR-r1i1p1','DMI-HIRHAM5-MPI-ESM-LR-r1i1p1', 'DMI-HIRHAM5-NorESM1-M-r1i1p1',
                     'GERICS-REMO2015-CNRM-CM5-r1i1p1', 'GERICS-REMO2015-CanESM2-r1i1p1','GERICS-REMO2015-EC-EARTH-r12i1p1', 'GERICS-REMO2015-HadGEM2-ES-r1i1p1','GERICS-REMO2015-IPSL-CM5A-MR-r1i1p1',
                     'GERICS-REMO2015-MIROC5-r1i1p1','GERICS-REMO2015-MPI-ESM-LR-r3i1p1', 'GERICS-REMO2015-NorESM1-M-r1i1p1','ICTP-RegCM4-6-CNRM-CM5-r1i1p1', 'ICTP-RegCM4-6-EC-EARTH-r12i1p1',
                     'ICTP-RegCM4-6-HadGEM2-ES-r1i1p1', 'ICTP-RegCM4-6-MPI-ESM-LR-r1i1p1', 'ICTP-RegCM4-6-NorESM1-M-r1i1p1', 'IPSL-WRF381P-CNRM-CM5-r1i1p1','IPSL-WRF381P-EC-EARTH-r12i1p1',
                     'IPSL-WRF381P-HadGEM2-ES-r1i1p1','IPSL-WRF381P-IPSL-CM5A-MR-r1i1p1', 'IPSL-WRF381P-MPI-ESM-LR-r1i1p1','IPSL-WRF381P-NorESM1-M-r1i1p1', 'KNMI-RACMO22E-CNRM-CM5-r1i1p1',
                     'KNMI-RACMO22E-EC-EARTH-r0i0p0', 'KNMI-RACMO22E-HadGEM2-ES-r1i1p1','KNMI-RACMO22E-IPSL-CM5A-MR-r1i1p1', 'KNMI-RACMO22E-MPI-ESM-LR-r1i1p1','KNMI-RACMO22E-NorESM1-M-r1i1p1',
                     'MOHC-HadREM3-GA7-05-CNRM-CM5-r1i1p1','MOHC-HadREM3-GA7-05-EC-EARTH-r12i1p1','MOHC-HadREM3-GA7-05-HadGEM2-ES-r1i1p1','MOHC-HadREM3-GA7-05-MPI-ESM-LR-r1i1p1','MOHC-HadREM3-GA7-05-NorESM1-M-r1i1p1',
                     'MPI-CSC-REMO2009-MPI-ESM-LR-r0i0p0', 'SMHI-RCA4-EC-EARTH-r0i0p0','SMHI-RCA4-HadGEM2-ES-r1i1p1', 'SMHI-RCA4-IPSL-CM5A-MR-r1i1p1','SMHI-RCA4-MPI-ESM-LR-r0i0p0', 'SMHI-RCA4-NorESM1-M-r1i1p1',
                     'UHOH-WRF361H-EC-EARTH-r1i1p1', 'UHOH-WRF361H-HadGEM2-ES-r1i1p1','UHOH-WRF361H-MIROC5-r1i1p1', 'UHOH-WRF361H-MPI-ESM-LR-r1i1p1']
        period = '2070/2099 - 1981/2010'

    # plot quadrant boundries (medians)
    plt.axvline(dsWi['tas_change'].median("member"),color='k',linewidth=1,linestyle='dashed',alpha=0.2)
    plt.axhline(dsWi['pr_change'].median("member"),color='k',linewidth=1,linestyle='dashed',alpha=0.2)
    for md in models:
        plt.scatter(dsWi['tas_change'].sel(member=md),dsWi['pr_change'].sel(member=md),**plot_kwargs[md],label=md)
    plt.xlabel('Target SAT Change, '+period+' (˚C)')
    min_sat = np.min(dsWi['tas_change'])
    max_sat = np.max(dsWi['tas_change'])
    plt.xlim([min_sat-0.5,max_sat+0.5])
    plt.ylabel('Target PR Change, '+period+' (mm/day)')
    min_pr = np.min(dsWi['pr_change'])
    max_pr = np.max(dsWi['pr_change'])
    plt.ylim([min_pr-0.5,max_pr+0.5])
    plt.title('SAT-PR Change',fontsize=12,fontweight='bold',loc='left')
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),fontsize=8)
    plt.savefig(plotname,bbox_inches='tight',dpi=300)
    plt.close()
