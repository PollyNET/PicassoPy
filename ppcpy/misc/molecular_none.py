


import numpy as np

def rayleigh_scattering(wavelength, pressure, temperature, C, rh):
    """
    RAYLEIGH_SCATTERING calculate the molecular volume backscatter coefficient and
    extinction coefficient.
    Inputs:
        wavelength: float
            Wavelength [nm]
        pressure: float
            The atmospheric pressure [hPa]
        temperature: float
            The atmospheric temperature [K]   
        C: float
            CO2 concentration [ppmv].
        rh: float
            Relative humidity from 0 to 100 [%] 
    Returns:
        beta_mol: float
            molecular backscatter coefficient. [m^{-1}*Sr^{-1}]
        alpha_mol: float
            molecular extinction coefficient. [m^{-1}]
    Reference:
        Bucholtz, A.: Rayleigh-scattering calculations for the terrestrial atmosphere, Appl. Opt. 34, 2765-2773 (1995)
        A. Behrendt and T. Nakamura, "Calculation of the calibration constant of polarization lidar and its dependency on atmospheric temperature," Opt. Express, vol. 10, no. 16, pp. 805-817, 2002.
    History:
        2017-12-16. First edition by Zhenping. All the code is based on the 
        python source code of Ioannis Binietoglou's [
        repo](https://bitbucket.org/iannis_b/lidar_molecular). 
        Detailed information please go to [repo](https://bitbucket.org/iannis_b/lidar_molecular)
    """
    beta_mol = beta_pi_rayleigh(wavelength, pressure, temperature, C, rh)
    alpha_mol = alpha_rayleigh(wavelength, pressure, temperature, C, rh)
    return beta_mol, alpha_mol

def sigma_pi_cabannes(wavelength, pressure, temperature, C, rh):
    """
    SIGMA_PI_CABANNES Calculate the backscattering cross section for the cabannes line. 
    Inputs:
        wavelength: float
            Light wavelength in nm
        pressure: float
            The atmospheric pressure [hPa]
        temperature: float
            The atmospheric temperature [K]   
        C: float
            CO2 concentration [ppmv].
        rh: float
            Relative humidity from 0 to 100 [%] 
    Returns:
        sigma:
            The backscattering cross section of the Cabannes line [m2sr-1].  
    """
    global ASSUME_AIR_IDEAL
    ASSUME_AIR_IDEAL = True

    p_e = rh_to_pressure(rh, temperature)
    epsilon = epsilon_atmosphere(wavelength, C, p_e, pressure)

    # Calculate properties of standard air
    n = n_air(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ASSUME_AIR_IDEAL)

    # Convert wavelength to meters
    lamda_m = wavelength * 10 ** -9

    # Separate in three factors for clarity
    f1 = 9 * np.pi ** 2 / (lamda_m ** 4 * N ** 2)
    f2 = (n ** 2 - 1) ** 2 / (n ** 2 + 2) ** 2
    f3 = 1 + 7 / 180. * epsilon
    sig_pi = f1 * f2 * f3
    return sig_pi

def beta_pi_cabannes(wavelength, pressure, temperature, C, rh):
    """
    BETA_PI_CABANNES Calculate the backscattering coefficient for the cabannes line. 
    Inputs:
        wavelength: float
            Light wavelength in nm
        pressure: float
            The atmospheric pressure [hPa]
        temperature: float
            The atmospheric temperature [K]
        C: float
            CO2 concentration [ppmv].
        rh: float
            Relative humidity from 0 to 100 [%] 
    Returns:
        beta_pi:
            The backscattering coefficient of the Cabannes line [m-1sr-1].  
    """
    global ASSUME_AIR_IDEAL
    ASSUME_AIR_IDEAL = True

    sigma_pi = sigma_pi_cabannes(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ASSUME_AIR_IDEAL)  # Number density of the atmosphere

    beta_pi = N * sigma_pi
    return beta_pi


