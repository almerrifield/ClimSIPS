
#################################
# packages
#################################

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["independence_square"]

def independence_square(outfile,cmip,im_or_em,season_region,plotname="independence_metric.png"):
    if cmip not in ['CMIP5','CMIP6','CH202x','CH202x_CMIP6','RCM']:
        raise NotImplementedError(cmip)
    if im_or_em not in ['IM','EM']:
        raise NotImplementedError(im_or_em)
    if season_region not in ['JJA_CEU','DJF_NEU','DJF_CEU','JJA_CH','DJF_CH','JJA_ALPS','DJF_ALPS']:
        raise NotImplementedError(season_region)


    ################################################
    fig = plt.figure(figsize=(13,10))
    ax = plt.subplot(111)
    ################################################
    dsWi = xr.open_dataset(outfile,use_cftime = True)

    # puts models in paper index order (TO DO: generalize)
    if cmip == 'CMIP6' and len(dsWi.delta_i) == 34:
        ind_order = [4,1,17,34,11,12,9,8,13,16,14,33,32,25,26,21,22,35,5,2,27,28,15,3,37,30,29,20,19,31,18,10,7,6]
    else:
        ind_order = np.arange(1,len(dsWi.delta_i)+1)
    if cmip == 'CMIP5' and len(dsWi.delta_i) == 26:
        ind_order = [1,2,6,7,12,28,26,17,15,16,20,21,3,10,9,8,19,18,14,13,27,5,4,23,22,25]
    else:
        ind_order = np.arange(1,len(dsWi.delta_i)+1)
    if cmip in ['CH202x','CH202x_CMIP6','RCM']:
        ind_order = np.arange(1,len(dsWi.delta_i)+1)

    dsWi = dsWi.assign_coords({"fam_order": ("member", ind_order)})
    dsWi = dsWi.assign_coords({"fam_order_2": ("member_model", ind_order)})
    dsWi_sort_a = dsWi.sortby(['fam_order'],ascending=False)
    dsWi_sort_b = dsWi_sort_a.sortby(['fam_order_2'],ascending=True)
    plt.pcolor(dsWi_sort_b['delta_i'],vmin=0,vmax=2.3,cmap='viridis')
    label_1 = dsWi_sort_b.member_model.data
    label_2 = dsWi_sort_b.member.data
    ax.set_xticks(np.arange(0.5,len(label_1)+0.5))
    ax.set_xticklabels(label_1,fontsize=9,rotation=90)
    ax.set_yticks(np.arange(0.5,len(label_2)+0.5))
    ax.set_yticklabels(label_2,fontsize=9)
    plt.title('Independence Metric',fontsize=13,fontweight='bold',loc='left')
    cbar = plt.colorbar()
    cbar.set_label('intermember distance',fontsize=10)
    fig.savefig(str(plotname),bbox_inches='tight',dpi=300)
    plt.close()
