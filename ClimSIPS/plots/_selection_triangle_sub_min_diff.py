###########################
# ternary plot
###########################
# need to enter the ternary environment
# qce
# conda activate ternary_plot

from pathlib import Path
import csv
import math
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import ternary

__all__ = ["selection_triangle_sub_min_diff"]

def shannon_entropy(p):
    """Computes the Shannon Entropy at a distribution in the simplex."""
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1.*s

def selection_triangle_sub_min_diff(optimal_models_csv,optimal_models_csv_min2,no_of_steps,plotname="cost_function_difference.png"):
    filename = optimal_models_csv

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        data = []
        for d in reader:
            d['alpha'] = np.round(float(d['alpha']), 3)
            d['beta'] = np.round(float(d['beta']), 3)
            nr_mem = len(d)-3
            d['models_str'] = ', '.join(sorted([d[f'member{i}'] for i in range(nr_mem)]))
            data.append(d)
# generalize d of member
    alphas = list(sorted(list(set([d['alpha'] for d in data]))))
    betas = list(sorted(list(set([d['beta'] for d in data]))))
    member_combo_data = [[np.nan for _ in betas] for _ in alphas]
    models_labels = [['' for _ in betas] for _ in alphas]
    models_costvals = [[0 for _ in betas] for _ in alphas]

    models_to_number = {}
    for d in data:
        ia, ib = alphas.index(d['alpha']),  betas.index(d['beta'])
        assert ia+ib <= no_of_steps # change here
        models_str = d['models_str']
        models_to_number.setdefault(models_str, len(models_to_number))
        member_combo_data[ia][ib] = models_to_number[models_str]
        models_labels[ia][ib] = models_str
        models_costvals[ia][ib] = float(d['min_val'])

    ds_test = xr.Dataset(dict(idx = (['alpha','beta'],member_combo_data), models=(['alpha','beta'],models_labels),costvals=(['alpha','beta'],models_costvals)), coords=dict(alpha=alphas,beta=betas))

    for k,v in models_to_number.items():
        print(v, k)

    data = {}
    for a in ds_test.alpha.data:
        for b in ds_test.beta.data:
            c = round(1-a-b,4)
            if c < 0:
                continue
            ia, ib = alphas.index(a),  betas.index(b)
            data[(ia, no_of_steps-ia-ib, ib)] = models_costvals[ia][ib] #change here


# cost val plot
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)

    tax = ternary.TernaryAxesSubplot(ax=ax, scale=100)
    # Remove default Matplotlib Axes
    tax.boundary(linewidth=1)
    # tax.gridlines(color="black", multiple=.5)
    tax.gridlines(color="w", multiple=10, linewidth=0.5)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()

    tax.ticks(axis='b', linewidth=1, multiple=10, tick_formats="%i%%", offset=.018, clockwise=True,fontsize=12)
    tax.ticks(axis='r', linewidth=1, multiple=10, tick_formats="%i%%", offset=.026, clockwise=True,fontsize=12)
    tax.ticks(axis='l', linewidth=1, multiple=10, tick_formats="%i%%", offset=.028, clockwise=True,fontsize=12)

    tax.left_axis_label(r"Performance ([1-$\alpha$-$\beta$] $\times$ 100%)", offset=.15,fontsize=14)
    tax.right_axis_label(r"Independence ($\alpha$ $\times$ 100%)", offset=.15,fontsize=14)
    tax.bottom_axis_label(r"Spread  ($\beta$ $\times$ 100%)",offset=0.05,fontsize=14)
    tax.heatmap(data, cmap='YlGnBu_r', vmin=-8,vmax=0, style="hexagonal") #36.5
    tax._redraw_labels()
    fig.tight_layout()
    fig.savefig(str("test_costval.png"),bbox_inches='tight',dpi=300)



#     fig = plt.figure(figsize=(13,6))
#     ax = fig.add_subplot(111)
#
#     tax = ternary.TernaryAxesSubplot(ax=ax, scale=no_of_steps)
#     # Remove default Matplotlib Axes
#     tax.boundary(linewidth=1)
#     nr_of_ticks=11
#     ticks = [i/(nr_of_ticks-1)*100 for i in range(nr_of_ticks)]
#     tax.gridlines(color="w", multiple=no_of_steps/(nr_of_ticks-1), linewidth=0.5)
#     tax.get_axes().axis('off')
#     tax.clear_matplotlib_ticks()
#
#     tax.ticks(ticks, axis='b', linewidth=1, multiple=no_of_steps/(nr_of_ticks-1), tick_formats="%i%%", offset=.018, clockwise=True,fontsize=12)
#     tax.ticks(ticks, axis='r', linewidth=1, multiple=no_of_steps/(nr_of_ticks-1), tick_formats="%i%%", offset=.026, clockwise=True,fontsize=12)
#     tax.ticks(ticks, axis='l', linewidth=1, multiple=no_of_steps/(nr_of_ticks-1), tick_formats="%i%%", offset=.028, clockwise=True,fontsize=12)
#     tax.set_axis_limits({key:(0,100) for key in 'lrb'})
#     tax.get_ticks_from_axis_limits(10)
#     tax.left_axis_label(r"Performance ([1-$\alpha$-$\beta$] $\times$ 100%)", offset=.15,fontsize=14)
#     tax.right_axis_label(r"Independence ($\alpha$ $\times$ 100%)", offset=.15,fontsize=14)
#     tax.bottom_axis_label(r"Spread  ($\beta$ $\times$ 100%)",offset=0.05,fontsize=14)
# ##################################################
#     ternary.heatmap(data, scale=no_of_steps, ax=ax, style="hexagonal", cmap=cmap_r, colorbar=False)
#     models_list = [k for k,v in sorted(models_to_number.items(), key=lambda it: it[1])]
#     vmin = min(data.values())
#     vmax = max(data.values())
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm._A = []
#     cb = plt.colorbar(sm, ax=ax)
#     cb.set_ticks([(len(models_list)-1)*(i-0.5)/len(models_list) for i in range(1,len(models_list)+1)])
#     models_list.reverse()
#     cb.ax.set_yticklabels(models_list,fontsize=10)
# ##################################################
#     tax._redraw_labels()
#     fig.tight_layout()
#     fig.savefig(str(plotname),bbox_inches='tight',dpi=300)
