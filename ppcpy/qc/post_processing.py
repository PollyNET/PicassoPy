import numpy as np
import xarray as xr
from scipy.signal import argrelmin, argrelmax
from numpy.lib.stride_tricks import sliding_window_view
import logging
import copy


def run_postprocesing(data_cube, nr:bool=False, retrieval="klett"):

    config_dict = data_cube.polly_config_dict

    height = data_cube.retrievals_highres["height"]

    channels = [(355, "total", "FR"),
                (532, "total", "FR"),
                (1064, "total", "FR")]
    if nr:
        channels += [(355, "total", "NR"),
                    (532, "total", "NR")]

    res = [{} for _ in data_cube.clFreeGrps]

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        for (wv, t, tel) in channels:
            logging.info(f"cldFree: {i} channel: {wv, t, tel} retrieval: {retrieval}")

            # add some checks if the data exist
            if not f"{wv}_{t}_{tel}" in data_cube.retrievals_profile[retrieval][i]:
                logging.info(f"skipping channel {wv}_{t}_{tel}")
                continue

            # Extract data
            d = copy.deepcopy(data_cube.retrievals_profile[retrieval][i][f"{wv}_{t}_{tel}"])

            res[i][f"{wv}_{t}_{tel}"]  = cut_overlap(config_dict, height, d, nr, retrieval)
            res[i][f"{wv}_{t}_{tel}"] = screening_abs_values(config_dict, res[i][f"{wv}_{t}_{tel}"], nr, retrieval)
            res[i][f"{wv}_{t}_{tel}"] = screening_rel_err(config_dict, res[i][f"{wv}_{t}_{tel}"], nr, retrieval)
            res[i][f"{wv}_{t}_{tel}"] = screening_abs_err(config_dict, res[i][f"{wv}_{t}_{tel}"], nr, retrieval)

    return res


def run_merging(data_cube, nr:bool=False, retrieval="klett_QC"):

    config_dict = data_cube.polly_config_dict
    
    height = data_cube.retrievals_highres["height"]
    
    channels = [(355, "total"),
                (532, "total")]

    res = [{} for _ in data_cube.clFreeGrps]

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        for (wv, t) in channels:
            logging.info(f"cldFree: {i} channel: {wv, t} retrieval: {retrieval}")

            # add some checks if the data exist
            if not f"{wv}_{t}_FR" in data_cube.retrievals_profile[retrieval][i]:
                logging.info(f"skipping channel {wv}_{t}_FR")

                if f"{wv}_{t}_NR" in data_cube.retrievals_profile[retrieval][i]:
                    logging.info(f"using only {wv}_{t}_NR")

                    res[i][f"{wv}_{t}_merged"] = data_cube.retrievals_profile[retrieval][i][f"{wv}_{t}_NR"]   
                continue

            if not f"{wv}_{t}_NR" in data_cube.retrievals_profile[retrieval][i]:
                logging.info(f"skipping channel {wv}_{t}_NR")

                res[i][f"{wv}_{t}_merged"] = data_cube.retrievals_profile[retrieval][i][f"{wv}_{t}_FR"]

            else:
                logging.info(f"successfully merged {wv}_{t}")
                # Extract data
                d_FR = copy.deepcopy(data_cube.retrievals_profile[retrieval][i][f"{wv}_{t}_FR"])
                d_NR = copy.deepcopy(data_cube.retrievals_profile[retrieval][i][f"{wv}_{t}_NR"])

                res[i][f"{wv}_{t}_merged"] = merge_NR_FR(config_dict, height, d_FR, d_NR, retrieval)

    return res

    
############################################################################################
### Overlap Cutting
############################################################################################
"""
def find_zero_2grad(variable):

    center = variable[1:-1]
    left = variable[:-2]
    right = variable[2:]

    valid_left = ~np.isnan(center) & ~np.isnan(left)
    valid_right = ~np.isnan(center) & ~np.isnan(right)

    left_diff  = (center != left)  & valid_left
    right_diff = (center != right) & valid_right

    # get indices, for which at least 1 of their neigbors has a different sign
    mask_2grad=np.logical_or(left_diff, right_diff)
    
    # mask starts at i=1 of the gradient (because of comparing with left + right neighbor)
    # true_idx1 are the original indices of the gradient array
    true_idx1 = np.where(mask_2grad)[0]+1

    # and take index[i], if index[i+1]=index[i]+1 to avoid double counting
    # true_idx2 are the indices within true_idx1
    # true_idx are again the original indices of the gradient array
    
    #does not yet work correctly, e.g., if i = 20, 21, 22, 23 with sign = -1, 1, 1, -1
    #only 20 and 22 should be selected, 21 is still selected
    #but if sign = -1, 1, -1, 1 selection of 21 would be correct then
    #but does not really matter, because only i.min() will be chosen and this should be correct
    
    true_idx2 = np.where(np.diff(true_idx1)==1)[0]
    true_idx = true_idx1[true_idx2]

    # neglect 1-digit indices
    min_idx = true_idx[true_idx>9] # is the same like true_idx[true_idx>9][0]
    min_id = min_idx.min()

    return min_id

############################################################################################
def cut_overlap_grad2(height, polly_ds, var_polly, nr_order):

    h = height
    grad2 = np.gradient(np.gradient(polly_ds[var_polly], h), h)
    #grad2 = np.gradient(np.gradient(polly_ds[var_polly]))
    grad2_sign = np.sign(grad2)
    min_id = find_zero_2grad(grad2_sign)
    #max1st_id = argrelmax(polly_ds[var_polly], order=nr_order)[0][0]

    return min_id
"""
############################################################################################
def cut_overlap_rel_grad(height, input_var, std_thrs, window_shape):

    h = height
    rel_grad = np.abs(np.gradient(np.gradient(input_var, h), h)/(np.gradient(input_var, h)))
    #rel_grad = np.abs(np.gradient(np.gradient(input_var))/(np.gradient(input_var)))
    lowest_non_nan_id = np.where(~np.isnan(rel_grad))[0][0]+2  # neglect first 2 height bins because there might be a wired step
    min_indices = lowest_non_nan_id + argrelmax(rel_grad[lowest_non_nan_id:len(rel_grad)-window_shape], order=2)[0]
        
    windows = sliding_window_view(rel_grad, window_shape=window_shape, axis=0)
    rel_std = np.nanstd(windows[min_indices], axis=1)
    min_id_cond = np.where(rel_std > std_thrs)[0][0]
    min_id_cond = min_indices[min_id_cond]

    return min_id_cond
    
############################################################################################
def cut_overlap(config_dict, height, polly_ds, NR, retrieval):

    ### ext
    if 'aerExt' in polly_ds and not np.isnan(polly_ds['aerExt']).all():
        ext_min_id = cut_overlap_rel_grad(height, polly_ds['aerExt'], config_dict['cutting_OL_std_thrs'], config_dict['cutting_OL_std_window_range'])

        polly_ds['aerExt'][:ext_min_id] = np.nan
        polly_ds['aerExtStd'][:ext_min_id] = np.nan
        polly_ds["ol_cut_height"]=height[ext_min_id]

        if 'AE_Ext_355_532' in polly_ds and not np.isnan(polly_ds['AE_Ext_355_532']).all():
            polly_ds['AE_Ext_355_532'][:ext_min_id] = np.nan
            polly_ds['AEStd_Ext_355_532'][:ext_min_id] = np.nan

        if 'AE_Ext_532_1064' in polly_ds and not np.isnan(polly_ds['AE_Ext_532_1064']).all():
            polly_ds['AE_Ext_532_1064'][:ext_min_id] = np.nan
            polly_ds['AEStd_Ext_532_1064'][:ext_min_id] = np.nan

    ### bsc
    if retrieval=='klett':
        if 'aerBsc' in polly_ds and not np.isnan(polly_ds['aerBsc']).all():
            #bsc_min_id = cut_overlap_rel_grad(height, polly_ds, 'aerBsc', config_dict['cutting_OL_std_thrs'], config_dict['cutting_OL_std_window_range'])
            bsc_min_id = ext_min_id

            polly_ds['aerBsc'][:bsc_min_id]=np.nan
            polly_ds['aerBscStd'][:bsc_min_id]=np.nan

            if 'AE_Bsc_355_532' in polly_ds and not np.isnan(polly_ds['AE_Bsc_355_532']).all():
                polly_ds['AE_Bsc_355_532'][:bsc_min_id] = np.nan
                polly_ds['AEStd_Bsc_355_532'][:bsc_min_id] = np.nan

            if 'AE_Bsc_532_1064' in polly_ds and not np.isnan(polly_ds['AE_Bsc_532_1064']).all():
                polly_ds['AE_Bsc_532_1064'][:bsc_min_id] = np.nan
                polly_ds['AEStd_Bsc_532_1064'][:bsc_min_id] = np.nan
        
    ### LR
    if retrieval=='raman':
        if 'LR' in polly_ds and not np.isnan(polly_ds['LR']).all():
            #lr_min_id = cut_overlap_rel_grad(polly_ds, lr_var_polly, config_dict['cutting_OL_std_thrs'])
            lr_min_id = ext_min_id

            polly_ds['LR'][:lr_min_id]=np.nan
            polly_ds['LRStd'][:lr_min_id]=np.nan

    ### depol
    if (NR==False) and (retrieval=='klett') and ('pdr' in polly_ds) and (not np.isnan(polly_ds['pdr']).all()):
        depol_min_id = bsc_min_id

        polly_ds['pdr'][:depol_min_id]=np.nan
        polly_ds['pdrStd'][:depol_min_id]=np.nan

        polly_ds['vdr'][:depol_min_id]=np.nan
        polly_ds['vdrStd'][:depol_min_id]=np.nan

    return polly_ds


############################################################################################
### Quality Screening of Polly Profiles
############################################################################################
def filter_for_nonnan_blocks(config_dict, polly_ds, var_polly):

    ### this is the filter function
    if var_polly in polly_ds:
        non_nan_mask = ~np.isnan(polly_ds[var_polly])
        diff = np.diff((np.concatenate(([False], non_nan_mask, [False]))).astype(int))
        block_starts = np.where(diff==1)[0]
        block_ends = np.where(diff==-1)[0]

        valid_mask = np.zeros_like(polly_ds[var_polly], dtype=bool)
    
        for start, end in zip(block_starts, block_ends):
             if end - start >=config_dict['nr_contiguous_nonnan']:
                  valid_mask[start:end] = True

        polly_ds[var_polly][~valid_mask] = np.nan

    return polly_ds


def filter_non_nan_blocks(config_dict, polly_ds, NR=False):

    ### here, the filter function is just applied in a loop to all parameters at once

    var_list = ['aerBsc', 'aerExt', 'LR', 'aerBscStd', 'aerExtStd', 'LRStd']
    for k in range(len(var_list)):
        polly_ds = filter_for_nonnan_blocks(config_dict, polly_ds, var_list[k])
    
    if NR==False:
        polly_ds = filter_for_nonnan_blocks(config_dict, polly_ds, 'pdr')
        polly_ds = filter_for_nonnan_blocks(config_dict, polly_ds, 'pdrStd')

    return polly_ds


############################################################################################
def screening_abs_values(config_dict, polly_ds, NR, retrieval):

    if retrieval=='raman' and 'LR' in polly_ds:
        polly_ds['LR'] = xr.where(polly_ds['aerBsc']*1e6 > config_dict['thrs_abs_values_bsc'], polly_ds['LR'], np.nan)
        polly_ds['LR'] = xr.where(polly_ds['aerExt']*1e6 > config_dict['thrs_abs_values_ext'], polly_ds['LR'], np.nan)
        polly_ds['LR'] = xr.where(polly_ds['LR'] >= 0, polly_ds['LR'], np.nan)
        polly_ds['LRStd'] = xr.where(polly_ds['aerBsc']*1e6 > config_dict['thrs_abs_values_bsc'], polly_ds['LRStd'], np.nan)
        polly_ds['LRStd'] = xr.where(polly_ds['aerExt']*1e6 > config_dict['thrs_abs_values_ext'], polly_ds['LRStd'], np.nan)
        polly_ds['LRStd'] = xr.where(polly_ds['LR'] >= 0, polly_ds['LRStd'], np.nan)
    
    if NR==False and 'pdr' in polly_ds:
        polly_ds['pdr'] = xr.where(polly_ds['aerBsc']*1e6 > config_dict['thrs_abs_values_bsc'], polly_ds['pdr'], np.nan)
        polly_ds['pdr'] = xr.where(polly_ds['pdr'] >=0, polly_ds['pdr'], np.nan)
        polly_ds['pdr'] = xr.where(polly_ds['pdr'] <=1, polly_ds['pdr'], np.nan)
        polly_ds['pdrStd'] = xr.where(polly_ds['aerBsc']*1e6 > config_dict['thrs_abs_values_bsc'], polly_ds['pdrStd'], np.nan)
        polly_ds['pdrStd'] = xr.where(polly_ds['pdr'] >=0, polly_ds['pdrStd'], np.nan)
        polly_ds['pdrStd'] = xr.where(polly_ds['pdr'] <=1, polly_ds['pdrStd'], np.nan)

    polly_ds = filter_non_nan_blocks(config_dict, polly_ds, NR)

    return polly_ds


def screening_rel_err(config_dict, polly_ds, NR, retrieval):

    if retrieval=='raman' and 'LR' in polly_ds:
        polly_ds['LR'] = xr.where(polly_ds['LRStd']/polly_ds['LR'] < config_dict['thrs_screening_rel_err'], polly_ds['LR'], np.nan)  
        polly_ds['LRStd'] = xr.where(polly_ds['LRStd']/polly_ds['LR'] < config_dict['thrs_screening_rel_err'], polly_ds['LRStd'], np.nan)  
    
    #if NR==False and 'pdr' in polly_ds:
    #    polly_ds['pdr'] = xr.where(polly_ds['pdrStd']/polly_ds['pdr'] < config_dict['thrs_screening_rel_err'], polly_ds['pdr'], np.nan)  
    #    polly_ds['pdrStd'] = xr.where(polly_ds['pdrStd']/polly_ds['pdr'] < config_dict['thrs_screening_rel_err'], polly_ds['pdrStd'], np.nan)  
    
    polly_ds = filter_non_nan_blocks(config_dict, polly_ds, NR)

    return polly_ds


def screening_abs_err(config_dict, polly_ds, NR, retrieval):

    if retrieval=='raman' and 'aerExt' in polly_ds:
        if NR==True:
            thrs_abs_err_ext = config_dict['thrs_abs_err_ext_NR']
            thrs_abs_err_lr = config_dict['thrs_abs_err_LR_NR']
        else:
            thrs_abs_err_ext = config_dict['thrs_abs_err_ext_FR']
            thrs_abs_err_lr = config_dict['thrs_abs_err_LR_FR']

        polly_ds['aerExt'] = xr.where(polly_ds['aerExtStd']*1e6 <= thrs_abs_err_ext, polly_ds['aerExt'], np.nan)
        polly_ds['aerExtStd'] = xr.where(polly_ds['aerExtStd']*1e6 <= thrs_abs_err_ext, polly_ds['aerExtStd'], np.nan)

        if 'LR' in polly_ds:   
            polly_ds['LR'] = xr.where(polly_ds['LRStd'] <= thrs_abs_err_lr, polly_ds['LR'], np.nan)
            polly_ds['LRStd'] = xr.where(polly_ds['LRStd'] <= thrs_abs_err_lr, polly_ds['LRStd'], np.nan)
    
    if NR==False and 'pdr' in polly_ds:
        polly_ds['pdr'] = xr.where(polly_ds['pdrStd'] <= config_dict['thrs_abs_err_depol'], polly_ds['pdr'], np.nan) 
        polly_ds['pdrStd'] = xr.where(polly_ds['pdrStd'] <= config_dict['thrs_abs_err_depol'], polly_ds['pdrStd'], np.nan) 

    polly_ds = filter_non_nan_blocks(config_dict, polly_ds, NR)

    return polly_ds


############################################################################################
### Merging NR and FR
############################################################################################
############################################################################################

def get_merging_id_diff_std_based(config_dict, height, ds_polly, ds_polly_NR, var_polly, h_thrs, std_thrs, default_id, mult):

    """
    h_thrs: above which height the NR should be cutted
    nr_std_bins: range within the std of the diff is calculated, at each height plus minus 0.5*nr_std_bins
    std_thrs: std should be smaller, otherwise the diff at that height is set to nan
    nr_smallest: the x smallest diff are selected
    """

    ### if the NR profile is not empty, calc difference of NR - FR
    ### otherwise, check if FR is empty and take the lowest non-nan height as merging height or set merging height to 0

    diff = np.abs(ds_polly_NR[var_polly]-ds_polly[var_polly])*mult

    ### diff profile might be empty if no overlap between NR + FR
    ### then, the default merging height is selected (for ext + LR uppermost (below height thrs) non-nan NR height if available, else 0)
    if not np.isnan(diff).all():
        ### neglect diff above height threshold
        mask_thrs = height < h_thrs
        diff[~mask_thrs] = np.nan

        ### if the diff profile is not empty, calc the std of the diff around each height bin
        ### and set the diff nan at all height bins where the std exceeds a threshold
        ### else: default height
        if not np.isnan(diff).all():

            ### central id are the indices belonging to windows_std
            ### these are also the correct ids of diff
            idx = np.arange(0,len(diff),1)
            windows_idx = sliding_window_view(idx, window_shape=config_dict['merging_window_std'], axis=0)
            central_id = (windows_idx.mean(axis=1)).astype(int)
                
            windows_diff = sliding_window_view(diff, window_shape=config_dict['merging_window_std'], axis=0)
            windows_diff = np.nan_to_num(windows_diff, nan=0.0)
            windows_std = windows_diff.std(axis=1)

            valid_mask = windows_std < std_thrs
            valid_idx = central_id[valid_mask]

            diff_masked = np.full_like(diff, np.nan, dtype=diff.dtype)
            diff_masked[valid_idx]=diff[valid_idx]

            ### if the diff profile is not empty, take the x smallest diff and select the one with the smallest std
            ### this is then the merging height
            ### else default merging height
            if not np.isnan(diff_masked).all():
                mask_non_nan = ~np.isnan(diff_masked)
                non_nan_indices_diff = np.where(mask_non_nan)[0]
                non_nan_values_diff = diff_masked[mask_non_nan]
                min_idx_diff = non_nan_indices_diff[np.argsort(non_nan_values_diff)][:config_dict['merging_nr_smallest']]
                ### ids in central_id and in std array are 0.5*(nr_std_bins-1) smaller than the original ids
                ### i.e., ids_in_central_id = min_idx_diff - 0.5*(nr_std_bins-1)
                ids_in_central_id = np.where(np.isin(central_id, min_idx_diff))[0]
                stds_of_nr_smallest = windows_std[ids_in_central_id]
                id_smallest_std = np.argmin(stds_of_nr_smallest)
                ### the offset of 0.5*(nr_std_bins-1) has to be added again
                min_id_diff = ids_in_central_id[id_smallest_std] + int(0.5*(config_dict['merging_window_std']-1))
            else:
                min_id_diff = default_id
                #print(var_polly, 'default (std empty)')

        else:
            min_id_diff = default_id
            #print(var_polly, 'default (after thrs)')
            diff_masked = diff
            
    else:
        min_id_diff = default_id
        #print(var_polly, 'default (before thrs)')
        diff_masked = diff


    return diff_masked, min_id_diff


############################################################################################
def calc_average_NR_FR(FR, NR):

    ### calc average of NR + FR wherever both are available
    ### and keep the values of NR or FR where only one of both is available
    ### nan where none of them is available

    result = np.where(~np.isnan(FR) & ~np.isnan(NR), 0.5*(FR+NR),
                      np.where(~np.isnan(FR), FR,
                               np.where(~np.isnan(NR), NR, np.nan)))
    
    return result


def smooth_merging_zone(config_dict, ds, var_polly, merge_id, x):

    ### gliding smoothing within the merging window
    ### not all values in there will be updated
    ### the more edge bins are missing, the larger the smoothing window is

    start_id_smooth = merge_id-2*x
    end_id_smooth = merge_id+2*x
    ### the larger the window size, the lager the starting bin of the smoothed result
    missing_bins = int(config_dict['merging_smoothing_window']/2)

    smoothing_zone = ds[var_polly][start_id_smooth:end_id_smooth+1]
    windows = sliding_window_view(smoothing_zone, window_shape=config_dict['merging_smoothing_window'], axis=0)
    smoothed_result = windows.mean(axis=1)

    ds[var_polly][start_id_smooth+missing_bins:end_id_smooth-missing_bins+1] = smoothed_result

    return ds


def merge_ds_with_average(config_dict, ds_polly, ds_polly_NR, var_polly, merge_id, x):

    """
    x: merging and smoothing zone
    smoothing_bins: window size for gliding smoothing
    """

    ds_polly[var_polly][:merge_id-x] = ds_polly_NR[var_polly][:merge_id-x]
    
    FR = ds_polly[var_polly][merge_id-x:merge_id+x+1]
    NR = ds_polly_NR[var_polly][merge_id-x:merge_id+x+1]
    ds_polly[var_polly][merge_id-x:merge_id+x+1] = calc_average_NR_FR(FR, NR)
    
    ds_polly[var_polly+'Std'][:merge_id-x] = ds_polly_NR[var_polly+'Std'][:merge_id-x]
    
    FR = ds_polly[var_polly+'Std'][merge_id-x:merge_id+x+1]
    NR = ds_polly_NR[var_polly+'Std'][merge_id-x:merge_id+x+1]
    ds_polly[var_polly+'Std'][merge_id-x:merge_id+x+1] = calc_average_NR_FR(FR, NR)

    ### avoid error when merging height is 0
    if merge_id > config_dict['merging_smoothing_window']:
        ds_polly = smooth_merging_zone(config_dict, ds_polly, var_polly, merge_id, x)
        ds_polly = smooth_merging_zone(config_dict, ds_polly, var_polly+'Std', merge_id, x)

    return ds_polly


############################################################################################
def merge_NR_FR(config_dict, height, ds_polly, ds_polly_NR, retrieval):

    ### bsc
    if 'aerBsc' in ds_polly and not np.isnan(ds_polly['aerBsc']).all() and not np.isnan(ds_polly_NR['aerBsc']).all():
        default_id_bsc = 0
        diff_bsc, merge_id_diff_bsc = get_merging_id_diff_std_based(config_dict, height, ds_polly, ds_polly_NR, 'aerBsc', config_dict['thrs_merging_h_bsc'], config_dict['thrs_merging_std_bsc'], default_id_bsc, 1e6)
        ds_polly_merged = merge_ds_with_average(config_dict, ds_polly, ds_polly_NR, 'aerBsc', merge_id_diff_bsc, config_dict['merging_smoothing_zone_bsc'])

    ### ext
    if 'aerExt' in ds_polly and not np.isnan(ds_polly['aerExt']).all() and not np.isnan(ds_polly_NR['aerExt']).all():
        mask_thrs = height < config_dict['thrs_merging_h_ext']
        ds_polly_NR['aerExt'][~mask_thrs] = np.nan
        default_id_ext = np.where(~np.isnan(ds_polly_NR['aerExt']))[0][-1]
    else:
        default_id_ext = 0
    
    diff_ext, merge_id_diff_ext = get_merging_id_diff_std_based(config_dict, height, ds_polly, ds_polly_NR, 'aerExt', config_dict['thrs_merging_h_ext'], config_dict['thrs_merging_std_ext'], default_id_ext, 1e6)
    ds_polly_merged = merge_ds_with_average(config_dict, ds_polly, ds_polly_NR, 'aerExt', merge_id_diff_ext, config_dict['merging_smoothing_zone_ext'])
    logging.info(f"ext {height[merge_id_diff_ext]}")
    
    ### LR
    if retrieval=='raman_QC' and 'LR' in ds_polly:
        #default_id_lr = merge_id_diff_ext
        if not np.isnan(ds_polly['LR']).all() and not np.isnan(ds_polly_NR['LR']).all():
            mask_thrs = height < config_dict['thrs_merging_h_lr']
            ds_polly_NR['LR'][~mask_thrs] = np.nan
            default_id_lr = np.where(~np.isnan(ds_polly_NR['LR']))[0][-1]
        else:
            default_id_lr = 0
        
        diff_lr, merge_id_diff_lr = get_merging_id_diff_std_based(config_dict, height, ds_polly, ds_polly_NR, 'LR', config_dict['thrs_merging_h_lr'], config_dict['thrs_merging_std_lr'], default_id_lr, 1)
        ds_polly_merged = merge_ds_with_average(config_dict, ds_polly, ds_polly_NR, 'LR', merge_id_diff_lr, config_dict['merging_smoothing_zone_lr'])    
        logging.info(f"LR {height[merge_id_diff_lr]}")
    else:
        merge_id_diff_lr = None
        logging.info('no LR')

        
    return ds_polly_merged


