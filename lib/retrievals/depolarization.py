
import logging
import numpy as np

from scipy.ndimage import uniform_filter1d
def smooth_signal(signal, window_len):
    return uniform_filter1d(signal, size=window_len, mode='nearest')

def voldepol_cldFreeGrps(data_cube, ret_prof_name):
    """
    """

    config_dict = data_cube.polly_config_dict
    opt_profiles = data_cube.data_retrievals[ret_prof_name]
    print('no_profiles ', len(opt_profiles))
    print(opt_profiles[0].keys())

    signal = 'TCor'
    signal = 'BGCor'

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1
        for channel in opt_profiles[i]:
            wv, t, tel = channel.split('_')
            flagt = data_cube.gf(wv, 'total', tel)
            flagc = data_cube.gf(wv, 'cross', tel)
            #indxt = np.where(flagt)[0]
            retrieval = opt_profiles[i][channel]['retrieval']
            if np.any(flagt) and np.any(flagc):
                logging.info(f'voldepol at channel {wv} cldFree {i} {cldFree}')

                sigt = np.nansum(np.squeeze(
                    data_cube.data_retrievals[f'sig{signal}'][slice(*cldFree),:,flagt]), axis=0)
                #bgt = np.nansum(np.squeeze(
                #    data_cube.data_retrievals[f'BG{signal}'][slice(*cldFree),data_cube.gf(wv, 'total', tel)]), axis=0)
                sigc = np.nansum(np.squeeze(
                    data_cube.data_retrievals[f'sig{signal}'][slice(*cldFree),:,flagc]), axis=0)

                print(channel, data_cube.pol_cali[int(wv)]['eta_best'])

                vdr, vdrStd = calc_profile_vdr(
                    sigt, sigc, config_dict['G'][flagt], config_dict['G'][flagc],
                    config_dict['H'][flagt], config_dict['H'][flagc],
                    data_cube.pol_cali[int(wv)]['eta_best'], config_dict[f'voldepol_error_{wv}'],
                    config_dict[f'smoothWin_{retrieval}_{wv}']
                    )
                opt_profiles[i][channel]['vdr'] = vdr
                opt_profiles[i][channel]['vdrStd'] = vdrStd

                mdr, mdrStd, flgaDeftMdr = get_MDR(
                    vdr, vdrStd, data_cube.refH[i][f"{wv}_{t}_{tel}"]['refHInd'],
                )
                if config_dict["flagUseTheoreticalMDR"]:
                    logging.info("use the theoretical MDR value")
                    mdr = data_cube.polly_default_dict[f"molDepol{wv}"]
                print(f"est. mdr {channel}  {mdr} {mdrStd}")
                opt_profiles[i][channel]['mdr'] = mdr
                opt_profiles[i][channel]['mdrStd'] = mdrStd
                
                # experimental code
                vdr, vdrStd = calc_profile_vdr(
                    sigt, sigc, config_dict['G'][flagt], config_dict['G'][flagc],
                    config_dict['H'][flagt], config_dict['H'][flagc],
                    data_cube.pol_cali[int(wv)]['eta_best'], config_dict[f'voldepol_error_{wv}'],
                    1
                    )
                mdr, mdrStd, flgaDeftMdr = get_MDR(
                    vdr, vdrStd, data_cube.refH[i][f"{wv}_{t}_{tel}"]['refHInd'],
                )
                print(f"est. mdr {channel}  {mdr} {mdrStd}  (smooth1)")

                print(opt_profiles[i][channel].keys())

    return opt_profiles


def calc_profile_vdr(sigt, sigc, Gt, Gr, Ht, Hr, eta, 
                     voldepol_error, window, flag_smooth_before=True):
#def polly_vdr_ghk(sig_tot, sig_cross, GT, GR, HT, HR, eta, 
#                  voldepol_error_a0, voldepol_error_a1, voldepol_error_a2, 
#                  smooth_window=1, flag_smooth_before=True):
    """Calculate volume depolarization ratio using GHK parameters.

    Parameters:
    ----------
    sigt : ndarray
        Signal strength of the total channel [photon count].
    sigc : ndarray
        Signal strength of the cross channel [photon count].
    Gt : float
        G parameter in the total channel.
    Gc : float
        G parameter in the cross channel.
    Ht : float
        H parameter in the total channel.
    Hc : float
        H parameter in the cross channel.
    eta : float
        Depolarization calibration constant.
    voldepol_error : float
        Systematic uncertainty coefficient (constant term).
        Systematic uncertainty coefficient (linear term).
        Systematic uncertainty coefficient (quadratic term).
    smooth_window : int, optional
        The width of the sliding smoothing window for the signal. Default is 1.
    flag_smooth_before : bool, optional
        Flag to control whether smoothing is applied before or after the signal ratio. Default is True.

    Returns:
    -------
    vol_depol : ndarray
        Volume depolarization ratio.
    vol_depol_std : ndarray
        Uncertainty of the volume depolarization ratio.

    References:
    ----------
    - Engelmann, R. et al. The automated multiwavelength Raman polarization and water-vapor lidar Polly XT: 
      the neXT generation. Atmospheric Measurement Techniques 9, 1767-1784 (2016).
    - Freudenthaler, V. et al. Depolarization ratio profiling at several wavelengths in pure Saharan dust 
      during SAMUM 2006. Tellus B 61, 165-179 (2009).
    - Freudenthaler, V. About the effects of polarising optics on lidar signals and the Delta90 calibration. 
      Atmos. Meas. Tech., 9, 4181–4255 (2016).

    History:
    -------
    - 2018-09-02: First edition by Zhenping
    - 2018-09-04: Change the smoothing order. Smoothing the signal ratio instead of smoothing the signal.
    - 2019-05-24: Add 'flag_smooth_before' to control the smoothing order.
    - 2024-08-13: MH: Change calculation to GHK parameters and eta as depolarization constant.

    """

    print(f"G {Gt} {Gr} H {Ht} {Hr} Eta {eta} error {voldepol_error} Window {window} ")
    # Smooth signals before or after ratio calculation
    if flag_smooth_before:
        sig_ratio = smooth_signal(sigc, window) / smooth_signal(sigt, window)
    else:
        sig_ratio = smooth_signal(sigc / sigt, window)

    # Calculate volume depolarization ratio using GHK parameters
    vol_depol = (sig_ratio / eta * (Gt + Ht) - (Gr + Hr)) / ((Gr - Hr) - sig_ratio / eta * (Gt - Ht))

    # Calculate systematic uncertainty
    vol_depol_std = (voldepol_error[0] + 
                     voldepol_error[1] * vol_depol + 
                     voldepol_error[1] * vol_depol**2)

    return vol_depol, vol_depol_std

def get_MDR(vdr, vdrStd, refHInd):
    """get the vdr at reference height

    in the matlab pollynet processing chain this is done by recalculating vdr for the
    reference height chunk 
    (Pollynet_Processing_Chain/lib/calibration/pollyMDRGHK.m)

    Assuming that it is more efficient to use the precalculated vdr

    The snr criterion is missing for this very first version
    """
    mdr = np.mean(vdr[slice(*refHInd)])
    mdrStd = np.mean(vdrStd[slice(*refHInd)])

    return mdr, mdrStd, False


def pardepol_cldFreeGrps(data_cube, ret_prof_name):
    """
    """

    config_dict = data_cube.polly_config_dict
    opt_profiles = data_cube.data_retrievals[ret_prof_name]
    print('no_profiles ', len(opt_profiles))
    print(opt_profiles[0].keys())
    signal = 'BGCor'

    for i, cldFree in enumerate(data_cube.clFreeGrps):
        cldFree = cldFree[0], cldFree[1] + 1
        for channel in opt_profiles[i]:
            wv, t, tel = channel.split('_')
            print(f"=== {channel } ==============================================================")
            flagt = data_cube.gf(wv, 'total', tel)
            flagc = data_cube.gf(wv, 'cross', tel)
            #indxt = np.where(flagt)[0]
            retrieval = opt_profiles[i][channel]['retrieval']
            if np.any(flagt) and np.any(flagc):
                logging.info(f'pardepol at channel {wv} cldFree {i} {cldFree}')

                pdr, pdrStd = calc_pdr(
                    opt_profiles[i][channel]['vdr'], opt_profiles[i][channel]['vdrStd'],
                    opt_profiles[i][channel]['aerBsc'], np.ones_like(opt_profiles[i][channel]['aerBscStd'])*1e-7,
                    data_cube.mol_profiles[f'mBsc_{wv}'][i,:], opt_profiles[i][channel]['mdr'],
                    opt_profiles[i][channel]['mdrStd'],
                )

                opt_profiles[i][channel]['pdr'] = pdr
                opt_profiles[i][channel]['pdrStd'] = pdrStd
                print(opt_profiles[i][channel].keys())

    return opt_profiles


def calc_pdr(vol_depol, vol_depol_std, aer_bsc, aer_bsc_std, mol_bsc, mol_depol, mol_depol_std):
    """Calculate the particle depolarization ratio and estimate its standard deviation.

    Parameters:
    ----------
    vol_depol : ndarray
        Volume depolarization ratio.
    vol_depol_std : ndarray
        Standard deviation of volume depolarization ratio.
    aer_bsc : ndarray
        Aerosol backscatter coefficient [m^{-1}Sr^{-1}].
    aer_bsc_std : ndarray
        Standard deviation of aerosol backscatter coefficient [m^{-1}Sr^{-1}].
    mol_bsc : ndarray
        Molecule backscatter coefficient [m^{-1}Sr^{-1}].
    mol_depol : float
        Molecule depolarization ratio, dependent on central wavelength and IF bandwidth.
    mol_depol_std : float
        Standard deviation of molecule depolarization ratio.

    Returns:
    -------
    par_depol : ndarray
        Particle depolarization ratio.
    par_depol_std : ndarray
        Standard deviation of particle depolarization ratio.

    References:
    ----------
    - Freudenthaler, V., et al., Depolarization ratio profiling at several wavelengths in pure Saharan dust during SAMUM 2006,
      Tellus B, 61, 165-179, 2009.

    History:
    -------
    - 2021-05-31: First edition by Zhenping

    """

    # Compute particle depolarization ratio
    par_depol = (vol_depol + 1) / (mol_bsc * (mol_depol - vol_depol) / aer_bsc / (1 + mol_depol) + 1) - 1

    # Compute partial derivatives using finite differences
    delta_vol_depol = 0.005
    delta_mol_depol = 0.0005
    delta_aer_bsc = 5e-8

    par_depol_vol_depol_func = lambda x: (x + 1) / (mol_bsc * (mol_depol - x) / aer_bsc / (1 + mol_depol) + 1) - 1
    deriv_par_depol_vol_depol = (par_depol_vol_depol_func(vol_depol + delta_vol_depol) - 
                                 par_depol_vol_depol_func(vol_depol)) / delta_vol_depol

    par_depol_mol_depol_func = lambda x: (vol_depol + 1) / (mol_bsc * (x - vol_depol) / aer_bsc / (1 + x) + 1) - 1
    deriv_par_depol_mol_depol = (par_depol_mol_depol_func(mol_depol + delta_mol_depol) - 
                                 par_depol_mol_depol_func(mol_depol)) / delta_mol_depol

    par_depol_aer_bsc_func = lambda x: (vol_depol + 1) / (mol_bsc * (mol_depol - vol_depol) / x / (1 + mol_depol) + 1) - 1
    deriv_par_depol_aer_bsc = (par_depol_aer_bsc_func(aer_bsc + delta_aer_bsc) - 
                               par_depol_aer_bsc_func(aer_bsc)) / delta_aer_bsc

    # Compute standard deviation
    par_depol_std = np.sqrt(deriv_par_depol_vol_depol**2 * vol_depol_std**2 +
                            deriv_par_depol_mol_depol**2 * mol_depol_std**2 +
                            deriv_par_depol_aer_bsc**2 * aer_bsc_std**2)

    return par_depol, par_depol_std
