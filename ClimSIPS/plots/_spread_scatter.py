#################################
# packages
#################################

import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import regionmask

import datetime
from scipy import signal, stats
from statsmodels.stats.weightstats import DescrStatsW
import xskillscore

__all__ = ["spread_scatter"]

def spread_scatter(filename,cmip,im_or_em,plotname="spread_scatter.png"):
    if cmip not in ['CMIP5','CMIP6']:
        raise NotImplementedError(cmip)
    if im_or_em not in ['IM','EM']:
        raise NotImplementedError(im_or_em)

    # ################################################
    fig = plt.figure(figsize=(9,7))
    ax = plt.subplot(111)
    # ################################################
    dsWi = xr.open_dataset(filename,use_cftime = True)

    # All keyword arguments for CMIP5 and CMIP6 (add here)
    plot_kwargs = {
        'ACCESS-CM2-r2i1p1f1': dict(c='tab:red',s=20,marker='x',alpha=1), # start CMIP6 IM
        'ACCESS-ESM1-5-r1i1p1f1': dict(c='tab:red',s=30,marker='+',alpha=1),
        'AWI-CM-1-1-MR-r1i1p1f1': dict(c='tab:orange',s=20,marker='x',alpha=1),
        'CAS-ESM2-0-r1i1p1f1': dict(c='tab:cyan',s=20,marker='x',alpha=1),
        'CESM2-WACCM-r2i1p1f1': dict(c='darkgoldenrod',s=20,marker='x',alpha=1),
        'CESM2-r11i1p1f1': dict(c='darkgoldenrod',s=20,marker='o',alpha=1),
        'CMCC-CM2-SR5-r1i1p1f1': dict(c='darkgoldenrod',s=20,marker='^',alpha=1),
        'CMCC-ESM2-r1i1p1f1': dict(c='darkgoldenrod',s=30,marker='*',alpha=1),
        'CNRM-CM6-1-HR-r1i1p1f2': dict(c='cornflowerblue',s=20,marker='x',alpha=1),
        'CNRM-CM6-1-r4i1p1f2': dict(c='cornflowerblue',s=20,marker='o',alpha=1),
        'CNRM-ESM2-1-r2i1p1f2': dict(c='cornflowerblue',s=20,marker='^',alpha=1),
        'CanESM5-r16i1p1f1': dict(c='dodgerblue',s=20,marker='x',alpha=1),
        'E3SM-1-1-r1i1p1f1': dict(c='k',s=20,marker='x',alpha=1),
        'FGOALS-f3-L-r1i1p1f1': dict(c='maroon',s=20,marker='x',alpha=1),
        'FGOALS-g3-r2i1p1f1': dict(c='maroon',s=20,marker='o',alpha=1),
        'GFDL-CM4-r1i1p1f1': dict(c='indigo',s=20,marker='x',alpha=1),
        'GFDL-ESM4-r1i1p1f1': dict(c='indigo',s=20,marker='o',alpha=1),
        'GISS-E2-1-G-r1i1p3f1': dict(c='blueviolet',s=20,marker='x',alpha=1),
        'HadGEM3-GC31-LL-r3i1p1f3': dict(c='tab:red',s=20,marker='o',alpha=1),
        'HadGEM3-GC31-MM-r1i1p1f3': dict(c='tab:red',s=20,marker='^',alpha=1),
        'INM-CM4-8-r1i1p1f1': dict(c='mediumseagreen',s=20,marker='x',alpha=1),
        'INM-CM5-0-r1i1p1f1': dict(c='mediumseagreen',s=20,marker='o',alpha=1),
        'IPSL-CM6A-LR-r6i1p1f1': dict(c='royalblue',s=20,marker='x',alpha=1),
        'KACE-1-0-G-r3i1p1f1': dict(c='tab:red',s=30,marker='*',alpha=1),
        'KIOST-ESM-r1i1p1f1': dict(c='darkslateblue',s=20,marker='x',alpha=1),
        'MIROC-ES2L-r1i1p1f2': dict(c='lightsalmon',s=20,marker='x',alpha=1),
        'MIROC6-r15i1p1f1': dict(c='lightsalmon',s=20,marker='o',alpha=1),
        'MPI-ESM1-2-HR-r1i1p1f1': dict(c='tab:orange',s=20,marker='o',alpha=1),
        'MPI-ESM1-2-LR-r10i1p1f1': dict(c='tab:orange',s=20,marker='^',alpha=1),
        'MRI-ESM2-0-r1i1p1f1': dict(c='palevioletred',s=20,marker='x',alpha=1),
        'NESM3-r1i1p1f1': dict(c='tab:orange',s=30,marker='*',alpha=1),
        'NorESM2-MM-r1i1p1f1': dict(c='darkgoldenrod',s=20,marker='d',alpha=1),
        'TaiESM1-r1i1p1f1': dict(c='darkgoldenrod',s=30,marker='+',alpha=1),
        'UKESM1-0-LL-r1i1p1f2': dict(c='tab:red',s=20,marker='d',alpha=1), # end CMIP6 IM
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
        'ACCESS1-0-r1i1p1': dict(c='tab:red',s=20,marker='x',alpha=1), # start CMIP5 IM
        'ACCESS1-3-r1i1p1': dict(c='tab:red',s=20,marker='o',alpha=1),
        'CCSM4-r6i1p1': dict(c='darkgoldenrod',s=20,marker='x',alpha=1),
        'CESM1-CAM5-r1i1p1': dict(c='darkgoldenrod',s=20,marker='o',alpha=1),
        'CNRM-CM5-r2i1p1': dict(c='cornflowerblue',s=20,marker='x',alpha=1),
        'CSIRO-Mk3-6-0-r10i1p1': dict(c='deeppink',s=20,marker='x',alpha=1),
        'CanESM2-r5i1p1': dict(c='dodgerblue',s=20,marker='x',alpha=1),
        'GFDL-CM3-r1i1p1': dict(c='indigo',s=20,marker='x',alpha=1),
        'GFDL-ESM2G-r1i1p1': dict(c='indigo',s=20,marker='o',alpha=1),
        'GFDL-ESM2M-r1i1p1': dict(c='indigo',s=20,marker='^',alpha=1),
        'GISS-E2-H-r1i1p3': dict(c='blueviolet',s=20,marker='x',alpha=1),
        'GISS-E2-R-r1i1p3': dict(c='blueviolet',s=20,marker='o',alpha=1),
        'HadGEM2-ES-r4i1p1': dict(c='tab:red',s=20,marker='^',alpha=1),
        'IPSL-CM5A-LR-r1i1p1': dict(c='royalblue',s=20,marker='x',alpha=1),
        'IPSL-CM5A-MR-r1i1p1': dict(c='royalblue',s=20,marker='o',alpha=1),
        'IPSL-CM5B-LR-r1i1p1': dict(c='royalblue',s=20,marker='^',alpha=1),
        'MIROC-ESM-r1i1p1': dict(c='lightsalmon',s=20,marker='x',alpha=1),
        'MIROC5-r3i1p1': dict(c='lightsalmon',s=20,marker='o',alpha=1),
        'MPI-ESM-LR-r2i1p1': dict(c='tab:orange',s=20,marker='x',alpha=1),
        'MPI-ESM-MR-r1i1p1': dict(c='tab:orange',s=20,marker='o',alpha=1),
        'MRI-CGCM3-r1i1p1': dict(c='palevioletred',s=20,marker='x',alpha=1),
        'NorESM1-M-r1i1p1': dict(c='darkgoldenrod',s=20,marker='^',alpha=1),
        'NorESM1-ME-r1i1p1': dict(c='darkgoldenrod',s=30,marker='*',alpha=1),
        'bcc-csm1-1-m-r1i1p1': dict(c='silver',s=20,marker='x',alpha=1),
        'bcc-csm1-1-r1i1p1': dict(c='silver',s=20,marker='o',alpha=1),
        'inmcm4-r1i1p1': dict(c='mediumseagreen',s=20,marker='x',alpha=1), # end CMIP5 IM
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
        'inmcm4-r1i1p1': dict(c='mediumseagreen',s=20,marker='x',alpha=1) # end CMIP5 EM
    }


    if cmip == 'CMIP6' and im_or_em == 'IM':
        models = ['ACCESS-CM2-r2i1p1f1','ACCESS-ESM1-5-r1i1p1f1','AWI-CM-1-1-MR-r1i1p1f1', 'CAS-ESM2-0-r1i1p1f1',
            'CESM2-WACCM-r2i1p1f1','CESM2-r11i1p1f1','CMCC-CM2-SR5-r1i1p1f1',
            'CMCC-ESM2-r1i1p1f1', 'CNRM-CM6-1-HR-r1i1p1f2','CNRM-CM6-1-r4i1p1f2','CNRM-ESM2-1-r2i1p1f2',
            'CanESM5-r16i1p1f1','E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1', 'FGOALS-g3-r2i1p1f1',
            'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1','GISS-E2-1-G-r1i1p3f1',
            'HadGEM3-GC31-LL-r3i1p1f3','HadGEM3-GC31-MM-r1i1p1f3', 'INM-CM4-8-r1i1p1f1', 'INM-CM5-0-r1i1p1f1',
            'IPSL-CM6A-LR-r6i1p1f1','KACE-1-0-G-r3i1p1f1','KIOST-ESM-r1i1p1f1','MIROC-ES2L-r1i1p1f2','MIROC6-r15i1p1f1',
            'MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-LR-r10i1p1f1','MRI-ESM2-0-r1i1p1f1','NESM3-r1i1p1f1',
            'NorESM2-MM-r1i1p1f1','TaiESM1-r1i1p1f1','UKESM1-0-LL-r1i1p1f2']
    if cmip == 'CMIP6' and im_or_em == 'EM':
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
    if cmip == 'CMIP5' and im_or_em == 'IM':
        models = ['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'CCSM4-r6i1p1',
            'CESM1-CAM5-r1i1p1','CNRM-CM5-r2i1p1', 'CSIRO-Mk3-6-0-r10i1p1',
            'CanESM2-r5i1p1', 'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1',
            'GFDL-ESM2M-r1i1p1', 'GISS-E2-H-r1i1p3','GISS-E2-R-r1i1p3','HadGEM2-ES-r4i1p1',
            'IPSL-CM5A-LR-r1i1p1','IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1',
            'MIROC-ESM-r1i1p1', 'MIROC5-r3i1p1','MPI-ESM-LR-r2i1p1',
            'MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1', 'NorESM1-M-r1i1p1',
            'NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1',
            'inmcm4-r1i1p1']
    if cmip == 'CMIP5' and im_or_em == 'EM':
        models = ['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'CCSM4-r0i0p0',
            'CESM1-CAM5-r0i0p0', 'CNRM-CM5-r0i0p0', 'CSIRO-Mk3-6-0-r0i0p0',
            'CanESM2-r0i0p0', 'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1',
            'GFDL-ESM2M-r1i1p1', 'GISS-E2-H-r0i0p0', 'GISS-E2-R-r0i0p0',
            'HadGEM2-ES-r0i0p0', 'IPSL-CM5A-LR-r0i0p0', 'IPSL-CM5A-MR-r1i1p1',
            'IPSL-CM5B-LR-r1i1p1', 'MIROC-ESM-r1i1p1', 'MIROC5-r0i0p0',
            'MPI-ESM-LR-r0i0p0', 'MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1',
            'NorESM1-M-r1i1p1', 'NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1',
            'bcc-csm1-1-r1i1p1', 'inmcm4-r1i1p1']

    # plot quadrant boundries (medians)
    plt.axvline(dsWi['tas_change'].median("member"),color='k',linewidth=1,linestyle='dashed',alpha=0.2)
    plt.axhline(dsWi['pr_change'].median("member"),color='k',linewidth=1,linestyle='dashed',alpha=0.2)
    for md in models:
        plt.scatter(dsWi['tas_change'].sel(member=md),dsWi['pr_change'].sel(member=md),**plot_kwargs[md],label=md)
    plt.xlabel('Target SAT Change (??C)')
    plt.xlim([0.5,6])
    plt.ylabel('Target PR Change (mm/day)')
    plt.ylim([-0.7,0.5])
    plt.title('SAT-PR Change',fontsize=12,fontweight='bold',loc='left')
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),fontsize=8)
    plt.savefig(plotname,bbox_inches='tight',dpi=300)
    plt.close()
