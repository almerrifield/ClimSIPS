#################################
# packages
#################################

import xarray as xr
import numpy as np

import xskillscore

import itertools
import math
import time
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from collections import defaultdict


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
    if CMIP == 'CH202x':
        members = csms.CMIP5_RCM_common_members
    if CMIP == 'CH202x_CMIP6':
        members = csms.CMIP6_RCM_common_members
    if CMIP == 'RCM':
        members = csms.RCM_common_members
    if CMIP == 'RCM_CMIP6':
        members = csms.RCM_common_members
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

# cosine-latitude weighted average for RCMs
def cos_lat_weighted_mean_xy(ds):
	weights = np.cos(np.deg2rad(ds.lat))
	weights.name = "weights"
	ds_weighted = ds.weighted(weights)
	weighted_mean = ds_weighted.mean(('y', 'x'))
	return weighted_mean

# compute rmse between model and observations
def compute_predictor_deltas(ds,ds_obs,key):
    if np.ndim(ds.lat) == 1:
        weights = [np.cos(np.deg2rad(ds.lat))]*len(ds.lon)
        weights = xr.concat(weights, "lon")
        weights['lon'] = ds.lon
        rmse = xskillscore.rmse(ds[key],ds_obs[key],dim=['lat','lon'],weights=weights,skipna=True)
    if np.ndim(ds.lat) == 2:
        coords=dict(x=("x", ds.x), y=("y", ds.y))
        ds = ds.assign_coords(coords)
        weights = np.cos(np.deg2rad(ds.lat))
        rmse = xskillscore.rmse(ds[key],ds_obs[key],dim=['x','y'],weights=weights,skipna=True)
    return rmse

# normalize performance predictors
def normalize_predictor_deltas(ds,ds_cat):
    ds_norm = ds/ds_cat.mean('member')
    return ds_norm

# compute performace metric
# need to be able to compute for an arbitrary number of predictors
def performance_metric(*ds_list):
    return sum(ds_list)/len(ds_list)

# choose ensemble mean or individual member
# TO DO: generalize ensemble mean for any base set

# get base model name, e.g. ACCESS-CM2
def get_model_base(name):
    return name.split('-r')[0]

# average over model base name ensembles
def average_and_maybe_rename(dss, members, new_member_name):
    if len(members) > 1:
        # Average over the 'member' dimension and set new member name
        averaged = dss.sel(member=members).mean('member')
        averaged = averaged.assign_coords(member=new_member_name)
    else:
        # For single member scenarios, just select the member without averaging
        averaged = dss.sel(member=members[0])  # No need to average if only one member
    return averaged


def ensemble_mean_or_individual_member(ds,choice,CMIP,season_region,spread_path,key=None):
    if key:
        dss = ds[key]
    else:
        dss = ds
    ## CMIP6 by ensemble mean
    if choice == 'EM':
        model_identifiers = dss.member.data
        # Grouping models by their base name
        models_grouped = defaultdict(list)
        for model in model_identifiers:
            base_name = get_model_base(model)
            models_grouped[base_name].append(model)
        results = []
        if CMIP == 'CMIP6':
            for model_name, members in models_grouped.items():
                new_member_name = f"{model_name}-r0i0p0f0" if len(members) > 1 else members[0]
                averaged_model = average_and_maybe_rename(dss, members, new_member_name)
                results.append(averaged_model)
        elif CMIP in ['CMIP5','CH202x','RCM']:
            for model_name, members in models_grouped.items():
                new_member_name = f"{model_name}-r0i0p0" if len(members) > 1 else members[0]
                averaged_model = average_and_maybe_rename(dss, members, new_member_name)
                results.append(averaged_model)
        # Concatenate all results along a new 'member' dimension
        ds_all = xr.concat(results, dim='member')
        return ds_all
    ## CMIP6 by spread
    if choice == 'IM':
        if CMIP == 'CMIP6':
            mem_out = csms.CMIP6_spread_maximizing_members(csms.CMIP6_common_members,season_region,spread_path)
            ds_all = dss.sel(member=mem_out)
            return ds_all.sortby(ds_all.member)
    ## CMIP5 by spread
        elif CMIP == 'CMIP5':
            mem_out = csms.CMIP5_spread_maximizing_members(csms.CMIP5_common_members,season_region,spread_path)
            ds_all = dss.sel(member=mem_out)
            return ds_all.sortby(ds_all.member)
    ## CMIP6 RCM by spread (for normalization; max warming patch)
        elif CMIP == 'CH202x_CMIP6':
            mem_out = csms.CMIP6_max_warming_members(csms.CMIP6_common_members,season_region,spread_path) ### patch here
            ds_all = dss.sel(member=mem_out)
            return ds_all.sortby(ds_all.member)
    ## CMIP5 RCM by spread (max warming patch)
        elif CMIP == 'CH202x':
            mem_out = csms.CMIP5_max_warming_members(csms.CMIP5_common_members,season_region,spread_path) ### patch here
            ds_all = dss.sel(member=mem_out)
            return ds_all.sortby(ds_all.member)
    ## RCM by spread (for normalization, current patch)
        elif CMIP == 'RCM_CMIP6':
            mem_out = csms.RCM_max_warming_members(csms.RCM_common_members,season_region,spread_path)
            ds_all = dss.sel(member=mem_out)
            return ds_all.sortby(ds_all.member)
    ## RCM by spread (max warming patch)
        elif CMIP == 'RCM':
            mem_out = csms.RCM_max_warming_members(csms.RCM_common_members,season_region,spread_path)
            ds_all = dss.sel(member=mem_out)
            return ds_all.sortby(ds_all.member)

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
    if np.ndim(ds.lat) == 1:
        weights = [np.cos(np.deg2rad(ds.lat))]*len(ds.lon)
        weights = xr.concat(weights, "lon")
        weights['lon'] = ds.lon
    if np.ndim(ds.lat) == 2:
        coords=dict(x=("x", ds.x), y=("y", ds.y))
        ds = ds.assign_coords(coords)
        weights = np.cos(np.deg2rad(ds.lat))
    mod_coords = ds.member.values
    nmod = len(mod_coords)
    res = xr.DataArray(np.empty(shape=(nmod, nmod)),
                        dims=("member", "member_model"), coords=dict(member=mod_coords, member_model=mod_coords))

    for mod1 in ds.transpose("member", ..., transpose_coords=False):
        for mod2 in ds.transpose("member", ..., transpose_coords=False):
            if np.ndim(ds.lat) == 1:
                a = xskillscore.rmse(mod1,mod2,dim=['lat','lon'],weights=weights,skipna=True)
            if np.ndim(ds.lat) == 2:
                a = xskillscore.rmse(mod1,mod2,dim=['x','y'],weights=weights,skipna=True)
            res.loc[dict(member=mod1.member, member_model=mod2.member)] = a

    return res.where(res!=0)

# normalize for independence
def normalize_independence_matrix(ds):
    return ds/np.nanmean(ds)

##################################################################
# functions for model ensemble subselection
##################################################################

# create csv with minimizing value and subset listed for each alpha-beta combo (one core)
def multi_run(m, cmip, im_or_em, season_region, alpha_steps, beta_steps, perf_cutoff, data, min2=False):
    min2_text=""
    if min2:
        min2_text='min2_'
    filename=Path(cmip+'_'+im_or_em+'_'+season_region+'_'+min2_text+'alpha-beta-scan.csv')
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
                min_val, min_member = single_run(m, alpha, beta, perf_cutoff, data, silent=True, min2=min2)
                print(alpha, beta, min_val, min_member)
                writer.writerow([alpha,beta,min_val]+min_member)
    return filename

# finds minimizing subset
def single_run(m, alpha, beta, perf_cutoff, data, silent=False, min2=False):
    n = len(data.delta_q)
    perf = xr.DataArray(data.delta_q.data, dims=['member'], coords=dict(member=data.delta_q.coords['member']))
    change_data = data.change.data
    dist_data = data.delta_i.data
    change = xr.DataArray(change_data, dims=['member','member_model'], coords=dict(member=data.change.coords['member'],member_model=data.change.coords['member_model']))
    dist = xr.DataArray(dist_data, dims=['member','member_model'], coords=dict(member=data.delta_i.coords['member'],member_model=data.delta_i.coords['member_model']))
    min_val, min_members = get_best_m_models(perf, dist, change, m, alpha, beta, perf_cutoff, silent=silent, min2=min2)
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
def get_best_m_models(perf, dist, change, m, alpha, beta, perf_cutoff, silent=True, min2=False):
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
    if min2:
        min2_val, min2_combo = np.inf, []

    total_combinations = int(math.factorial(n) / math.factorial(m) / math.factorial(n-m))
    start_time = time.time()

    for i, combo in enumerate(itertools.combinations(range(len(members)), m)):
        cost = cost_function(list(combo))
        if cost < min_val:
            if min2:
                min2_val = min_val
                min2_combo = min_combo
            min_combo = combo
            min_val = cost.data
        elif min2 and cost < min2_val:
            min2_combo = combo
            min2_val = cost.data

        minX_val = min_val
        minX_combo = min_combo
        if min2:
            minX_val = min2_val
            minX_combo = min2_combo
        minX_members = [members[i] for i in minX_combo]

        # this part displays progress, requires silent = False
        if not silent and i & 0b1111111111111 == 0:
            if i == 0:
                continue
            percent = i / total_combinations
            eta = (1-percent) * (time.time() - start_time) / percent
            print(f"{100*percent:>4.1f}% / eta in {eta/60:.1f} min / best score {minX_val:.3f}")
            print(f"{', '.join(minX_members )}")

    if not silent:
        print(f"all {total_combinations} combinations tested, which took {(time.time() - start_time)/60:.1f} min")
        print(f"min val (alpha={alpha}): {minX_val}")
        print(f"min members:")
        for index, member in zip(minX_combo, minX_members):
            distances = [f"{dist[index, i].data:>6.2f}" for i in minX_combo]
            spreads = [f"{change[index, i].data:>6.2f}" for i in minX_combo]
            print(f" * {member:>24}   perf: {perf[index].data:>6.2f} dist: {' '.join(distances)} spread: {' '.join(spreads)}") # avr_dist: {avr_dist[index].data:>6.2f}
    return minX_val, minX_members

# creates csv in parallel (when multiple cores are available)
def multi_parallel_run(m, cmip, im_or_em, season_region, alpha_steps, beta_steps, perf_cutoff, data, max_workers, min2=False):
    print(f'running with {max_workers} workers.')
    min2_text=""
    if min2:
        min2_text='min2_'
    single_run_subdir = cmip+'_'+season_region+'_'+min2_text+im_or_em
    filename=Path(cmip+'_'+im_or_em+'_'+season_region+'_'+min2_text+'alpha-beta-scan.csv')
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
                future = pool.submit(single_run_with_save, single_run_file, m, alpha, beta, perf_cutoff, data, silent=True, min2=min2)
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
def single_run_with_save(filename, m, alpha, beta, perf_cutoff, data, silent=False, min2=False):
    min_val, min_member = single_run(m, alpha, beta, perf_cutoff, data, silent=True, min2=min2)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([alpha,beta,min_val]+min_member)

# ################################
# Make output files
# ################################

## TO DO: pass performance CSV file
## - set new CMIP6 common members

def make_output_file(dsDeltaQ,ds_spread_metric,targets,dsWi,outfile='perf_ind_spread_metrics.nc'):
    print('Models Used: ',dsWi.member.data)
    dsWi = dsWi.to_dataset(name='delta_i')
    dsWi['delta_q'] = dsDeltaQ
    dsWi['change'] = ds_spread_metric
    dsWi['tas_change'] = targets[0]
    dsWi['pr_change'] = targets[1]
    dsWi.to_netcdf(outfile)

def select_models(outfile, cmip, im_or_em, season_region, m, alpha_steps, beta_steps, perf_cutoff,max_workers=1, min2=False):
    data = xr.open_dataset(outfile,use_cftime = True)
    if max_workers==1:
        return multi_run(m, cmip, im_or_em, season_region, alpha_steps, beta_steps, perf_cutoff, data, min2=min2)
    else:
        return multi_parallel_run(m, cmip, im_or_em, season_region, alpha_steps, beta_steps, perf_cutoff, data, max_workers, min2=min2)
