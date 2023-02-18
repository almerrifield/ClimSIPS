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

__all__ = ["performance_order"]

def performance_order(outfile,cmip,im_or_em,plotname="performance_order.png"):
    if cmip not in ['CMIP5','CMIP6']:
        raise NotImplementedError(cmip)
    if im_or_em not in ['IM','EM']:
        raise NotImplementedError(im_or_em)

    ################################################
    fig = plt.figure(figsize=(8,4))
    ax = plt.subplot(111)
    ################################################
    dsWi = xr.open_dataset(outfile,use_cftime = True)

    dsWi = dsWi.assign_coords({"perf": ("member", dsWi.delta_q)})
    dsWi_sort = dsWi.sortby(['perf'],ascending=True)

    ind = np.arange(0,len(dsWi_sort.delta_q)*2,2)
    for ii in ind:
        tickwid = [-0.5+ii,0.5+ii]
        plt.plot(tickwid,[dsWi_sort.delta_q.isel(member=int(ii/2)),dsWi_sort.delta_q.isel(member=int(ii/2))],linewidth=2,color='k')
        plt.xlim([-1,ind[-1]+1])
        plt.ylim([0.5,2])
        xticks = ind
        ax.set_xticks(xticks)
        labels = dsWi_sort.member.data
        ax.set_xticklabels(labels,fontsize=8,rotation = 90)
        plt.ylabel('Aggregate Distance from Observed',fontsize=10)
        plt.title('Performance Metric' ,fontsize=12,fontweight='bold',loc='left')
        plt.savefig(plotname,bbox_inches='tight',dpi=300)
