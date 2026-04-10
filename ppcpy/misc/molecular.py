
from collections import defaultdict
import numpy as np
import xarray as xr

# Global variable
ASSUME_AIR_IDEAL:bool = True


def air_refractive_index(wavelength:float, pressure:np.ndarray, temperature:np.ndarray, 
                         C:float, relative_humidity:np.ndarray) -> np.ndarray:
    """Calculate the refractive index of air.

    Parameters
    ----------
    wavelength : float
        Wavelength [nm].
    pressure : ndarray
        Atmospheric pressure [hPa].
    temperature : ndarray
        Atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    relative_humidity : ndarray
        Relative humidity [%].

    Returns
    -------
    n_air : ndarray
        Refractive index of air.
    """

    Xw = molar_fraction_water_vapour(pressure, temperature, relative_humidity)

    rho_axs = moist_air_density(1013.25, 288.15, C, 0)[0]
    rho_ws = moist_air_density(13.33, 293.15, 0, 1)[0]

    _, rho_a, rho_w = moist_air_density(pressure, temperature, C, Xw)

    n_axs = n_standard_air_with_CO2(wavelength, C)
    n_ws = n_water_vapor(wavelength)

    n_air = 1 + (rho_a / rho_axs) * (n_axs - 1) + (rho_w / rho_ws) * (n_ws - 1)
    return n_air


def moist_air_density(pressure:float|np.ndarray, temperature:float|np.ndarray, C:float, 
                      Xw:float|np.ndarray) -> tuple:
    """Calculate the density of moist air.

    Parameters
    ----------
    pressure : float or ndarray
        Total pressure [hPa].
    temperature : float or ndarray
        Temperature [K].
    C : float
        CO2 concentration [ppmv].
    Xw : float or ndarray
        Molar fraction of water vapor.
    
    Returns
    -------
    rho : float or ndarray
        ...
    rho_air : float or ndarray
        ...
    rho_wv : float or ndarray
        ...
    
    Notes
    -----
    .. TODO:: Finish docstring.
    """

    const = physical_constants()

    Ma = molar_mass_dry_air(C)
    Mw = 0.018015

    Z = compressibility_of_moist_air(pressure, temperature, Xw)

    P = pressure * 100.
    T = temperature

    rho = P * Ma / (Z * const['R'] * T) * (1 - Xw * (1 - Mw / Ma))
    rho_air = (1 - Xw) * P * Ma / (Z * const['R'] * T)
    rho_wv = Xw * P * Mw / (Z * const['R'] * T)

    return rho, rho_air, rho_wv


def molar_mass_dry_air(C:float) -> float:
    """Molar mass of dry air as a function of CO2 concentration.

    Parameters
    ----------
    C : float
        CO2 concentration [ppmv].

    Returns
    -------
    Ma : float
        Molar mass of dry air [kg/mol].
    """
    C1 = 400.
    Ma = 10 ** -3 * (28.9635 + 12.011e-6 * (C - C1))
    return Ma


def compressibility_of_moist_air(pressure:float|np.ndarray, temperature:float|np.ndarray, 
                                 molar_fraction:float|np.ndarray) -> float|np.ndarray:
    """Compressibility of moist air.

    Parameters
    ----------
    pressure : float or ndarray
        Total pressure [hPa].
    temperature : float or ndarray
        Temperature [K].
    molar_fraction : float or ndarray
        Molar fraction of water vapor.
    
    Returns
    -------
    Z : float or ndarray
        ...
    
    Notes
    -----
    .. TODO:: Finish docstring.
    """
    a0 = 1.58123e-6
    a1 = -2.9331e-8
    a2 = 1.1043e-10
    b0 = 5.707e-6
    b1 = -2.051e-8
    c0 = 1.9898e-4
    c1 = -2.376e-6
    d0 = 1.83e-11
    d1 = -7.65e-9

    p = pressure * 100.
    T = temperature
    Tc = temperature - 273.15

    Xw = molar_fraction

    Z = 1 - (p / T) * (a0 + a1 * Tc + a2 * Tc ** 2 + (b0 + b1 * Tc) * Xw +
                       (c0 + c1 * Tc) * Xw ** 2) + (p / T) ** 2 * (d0 + d1 * Xw ** 2)
    return Z


def n_standard_air(wavelength:float) -> float:
    """Calculate the refractive index of standard air at a given wavelength.

    Parameters
    ----------
    wavelength : float
        Wavelength [nm].

    Returns
    -------
    ns : float
        Refractive index of standard air.
    """
    wl_micrometers = wavelength / 1000.0
    s = 1 / wl_micrometers
    c1 = 5792105.
    c2 = 238.0185
    c3 = 167917.
    c4 = 57.362
    ns = 1 + (c1 / (c2 - s ** 2) + c3 / (c4 - s ** 2)) * 1e-8
    return ns


def n_standard_air_with_CO2(wavelength:float, C:float) -> float:
    """Calculate the refractive index of air at a specific wavelength with CO2 concentration.

    Parameters
    ----------
    wavelength : float
        Wavelength [nm].
    C : float
        CO2 concentration [ppmv].

    Returns
    -------
    n_axs : float
        Refractive index of air for the given CO2 concentration.
    """
    C2 = 450.
    n_as = n_standard_air(wavelength)
    n_axs = 1 + (n_as - 1) * (1 + 0.534e-6 * (C - C2))
    return n_axs


def n_water_vapor(wavelength:float) -> float:
    """Calculate the refractive index of water vapor.

    Parameters
    ----------
    wavelength : float
        Wavelength [nm].

    Returns
    -------
    n_ws : float
        Refractive index of water vapor.
    """
    wl_micrometers = wavelength / 1000.0
    s = 1 / wl_micrometers
    c1 = 1.022
    c2 = 295.235
    c3 = 2.6422
    c4 = 0.032380
    c5 = 0.004028
    n_ws = 1 + c1 * (c2 + c3 * s ** 2 - c4 * s ** 4 + c5 * s ** 6) * 1e-8
    return n_ws


def alpha_rayleigh(wavelength:float, pressure:float|np.ndarray, temperature:float|np.ndarray,
                   C:float, rh:float|np.ndarray) -> float|np.ndarray:
    """Cacluate the extinction coefficient for Rayleigh scattering. 

    Parameters
    ----------
    wavelength : float
        Wavelegnth [nm].
    pressure : float or ndarray
        Atmospheric pressure [hPa].
    temperature : float or ndarray
        Atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    rh : float or ndarray
        Relative humidity from 0 to 100 [%].

    Returns
    -------
    alpha: float or ndarray
        The molecular scattering coefficient [m^{-1}].
    """
    ASSUME_AIR_IDEAL = True
    sigma = sigma_rayleigh(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ASSUME_AIR_IDEAL)
    alp = N * sigma
    return alp


def beta_pi_rayleigh(wavelength:float, pressure:float|np.ndarray, temperature:float|np.ndarray, 
                     C:float, rh:float|np.ndarray) -> float|np.ndarray:
    """Calculates the total Rayleigh backscatter coefficient.

    Parameters
    ----------
    wavelength : float
        Wavelength [nm].
    pressure : float or ndarray
        The atmospheric pressure [hPa].
    temperature : float or ndarray
        The atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    rh : float or ndarray
        Relative humidity from 0 to 100 [%].

	Returns
    -------
    beta_pi : float or ndarray
        molecule backscatter coefficient [m^{-1}Sr^{-1}].
    """
    ASSUME_AIR_IDEAL = True
    dsigma_pi = dsigma_phi_rayleigh(np.pi, wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ASSUME_AIR_IDEAL)
    beta_pi = dsigma_pi * N
    return beta_pi


def dsigma_phi_rayleigh(theta:float, wavelength:float, pressure:float|np.ndarray,
                        temperature:float|np.ndarray, C:float, rh:float|np.ndarray) -> float|np.ndarray:
    """Calculates the angular rayleigh scattering cross section per molecule.

    Parameters
    ----------
    theta : float
        Scattering angle [rads].
    wavelength : float
        Wavelength [nm].
    pressure : float or ndarray
        The atmospheric pressure [hPa].
    temperature : float or ndarray
        The atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    rh : float or ndarray
        Relative humidity from 0 to 100 [%].

    Returns
    -------
    dsigma : float or ndarray
        rayleigh-scattering cross section [m2sr-1].
    """
    phase = phase_function(theta, wavelength, pressure, temperature, C, rh)
    phase = phase / (4 * np.pi)
    sigma = sigma_rayleigh(wavelength, pressure, temperature, C, rh)
    dsig = sigma * phase
    return dsig


def enhancement_factor_f(pressure:float, temperature:float) -> float:
    """Enhancement factor.

    Parameters
    ----------
    pressure : float or ndarray
        Atmospheric pressure [hPa].
    temperature : float or ndarray
        Atmospehric temperature [K].
    
    Returns
    -------
    f : float or ndarray
        Enhancment factor.
    """
    T = temperature
    p = pressure * 100.
    f = 1.00062 + 3.14e-8 * p + 5.6e-7 * (T - 273.15) ** 2
    return f


def kings_factor_atmosphere(wavelength:float, C:float, p_e:float|np.ndarray, 
                            p_t:float|np.ndarray) -> float|np.ndarray:
    """Calculate the king factor.

    Parameters
    ----------
    wavelength : float
        Wavelength [nm].
    C : float
        CO2 concentration [ppmv].
    p_e:  float or ndarray
        Water vapor pressure [hPa].
    p_t : float or ndarray
        Total air pressure [hPa].

    Returns
    -------
    k : float or ndarray
        Total atmospheric King's factor.

    References
    ----------
    https://bitbucket.org/iannis_b/lidar_molecular

    Examples
    --------
    k = king_factor_atmosphere(wavelength, C, p_e, p_t)
    """
    if not (200 < wavelength < 4000):
        raise ValueError('King\'s factor formula is only valid from 0.2 to 4um.')

    lamda_cm = wavelength * 10 ** -7
    wavenumber = 1 / lamda_cm

    F_N2 = kings_factor_N2(wavenumber)
    F_O2 = kings_factor_O2(wavenumber)
    F_ar = kings_factor_Ar()
    F_CO2 = kings_factor_CO2()
    F_H2O = kings_factor_H2O()

    c_n2 = 0.78084
    c_o2 = 0.20946
    c_ar = 0.00934
    c_co2 = 1e-6 * C
    c_h2o = p_e / p_t

    c_tot = c_n2 + c_o2 + c_ar + c_co2 + c_h2o

    k = (c_n2 * F_N2 + c_o2 * F_O2 + c_ar * F_ar + c_co2 * F_CO2 + c_h2o * F_H2O) / c_tot
    return k


def kings_factor_N2(wavenumber:float) -> float:
    """Approximates the King's correction factor of N2 for a specific wavenumber.
	According to Bates, the agreement with experimental values is "rather better than 1 per cent."

    Parameters
    ----------
    wavenumber : float
        Wavenumber (defined as 1/lamda) [cm^{-1}].

    Returns
    -------
    k : float
        Kings factor for N2.

    Notes
    -----
    The King's factor is estimated as

    .. math:: F_{N_2} = 1.034 + 3.17 \\cdot 10^{-4} \\cdot \\lambda^{-2}

    where :math:`\\lambda` is the wavelength in micrometers.
    

	References
    ----------
    Tomasi, C., Vitale, V., Petkov, B., Lupi, A. & Cacciari, A. Improved
    algorithm for calculations of Rayleigh-scattering optical depth in standard
    atmospheres. Applied Optics 44, 3320 (2005).
    Bates, D. R.: Rayleigh scattering by air, Planetary and Space Science, 32(6),
    785-790, doi:10.1016/0032-0633(84)90102-8, 1984.
    """
    lamda_cm = 1 / wavenumber
    lamda_um = lamda_cm * 10 ** 4
    k = 1.034 + 3.17e-4 * lamda_um ** -2
    return k


def kings_factor_O2(wavenumber:float):
    """Approximates the King's correction factor of O2 for a specific wavenumber.
	According to Bates, the agreement with experimental values is "rather better than 1 per cent."

    Parameters
    ----------
    wavenumber : float
        Wavenumber (defined as 1/lamda) [cm^{-1}].

    Returns
    -------
    k : float
        Kings factor for O2.
    
    Notes
    -----
    The King's factor is estimated as

    .. math:: F_{O_2} = 1.096 + 1.385 \\cdot 10^{-3} \\cdot \\lambda^{-2} + 1.448 \\cdot 10^{-4} \\cdot \\lambda^{-4}

    where :math:`\\lambda` is the wavelength in micrometers.

    References
    ----------
    Tomasi, C., Vitale, V., Petkov, B., Lupi, A. & Cacciari, A. Improved
    algorithm for calculations of Rayleigh-scattering optical depth in standard
    atmospheres. Applied Optics 44, 3320 (2005).
    Bates, D. R.: Rayleigh scattering by air, Planetary and Space Science, 32(6),
    785-790, doi:10.1016/0032-0633(84)90102-8, 1984.
    """
    lamda_cm = 1 / wavenumber
    lamda_um = lamda_cm * 10 ** 4
    k = 1.096 + 1.385e-3 * lamda_um ** -2 + 1.448e-4 * lamda_um ** -4
    return k


def kings_factor_Ar() -> float:
    """King's correction factor for Ar.
    
    Returns
    -------
    float
        Kings factor for Ar.
    """
    return 1.0


def kings_factor_CO2() -> float:
    """King's correction factor for CO2.
    
    Returns
    -------
    float
        Kings factor for CO2.
    """
    return 1.15


def kings_factor_H2O() -> float:
    """King's correction factor for H2O.
    
    Returns
    -------
    float
        Kings factor for H2O.
    """
    return 1.001


def molar_fraction_water_vapour(pressure:float|np.ndarray, temperature:float|np.ndarray, relative_humidity:float|np.ndarray) -> float|np.ndarray:
    """Molar fraction of water vapor. 

    Parameters
    ----------
    pressure : float or ndarray
        Total pressure [hPa].
    temperature : float or ndarray
        Atmospehric temperature [K] .
    relative_humidity : float or ndarray
        Relative humidity from 0 to 100 [%].
    
    Returns
    Xw : float or ndarray
        Molar fraction of water vapor.
    """
    p = pressure
    h = relative_humidity / 100.
    f = enhancement_factor_f(pressure, temperature)
    svp = saturation_vapor_pressure(temperature)
    p_wv = h * f * svp
    Xw = p_wv / p
    return Xw


def number_density_at_pt(pressure:float|np.ndarray, temperature:float|np.ndarray, 
                         relative_humidity:float|np.ndarray, ideal:bool) -> float|np.ndarray:
    """Calculate the number density for a given temperature and pressure, 
    taking into account the compressibility of air.

    Parameters
    ----------
    pressure : float or array
        Pressure [hPa].
    temperature : float or array
        Temperature [K].
    relative_humidity: float or array (?)
        ? The relative humidity of air (Check)
    ideal: bool
        If False, the compressibility of air is considered. If True, the 
        compressibility is set to 1.

    Returns
    -------
    n : float or array
        Number density of the atmosphere [m^{-3}].
    """
    Xw = molar_fraction_water_vapour(pressure, temperature, relative_humidity)
    if ideal:
        Z = 1
    else:
        Z = compressibility_of_moist_air(pressure, temperature, Xw)
    p_pa = pressure * 100.
    const = physical_constants()
    n = p_pa / (Z * temperature * const['k_b'])
    return n


def phase_function(theta:float, wavelength:float, pressure:float|np.ndarray, 
                   temperature:float|np.ndarray, C:float, rh:float|np.ndarray) -> float|np.ndarray:
    """Calculates the phase function at an angle theta for a specific wavelegth.

    Parameters
    ----------
    theta : float
        Scattering angle [rads].
    wavelength : float
        Wavelength [nm].
    pressure : float or ndarray
        The atmospheric pressure [hPa].
    temperature : float or ndarray
        The atmospheric temperature [K].  
    C : float
        CO2 concentration [ppmv].
    rh : float or ndarray
        Relative humidity from 0 to 100 [%]. 
        
	Returns
    -------
    p : float or ndarray
        Scattering phase function.
        
    Notes
    -----
    The formula is derived from Bucholtz (1995). A different formula is given in 
    Miles (2001). 

    The use of this formula insetad of the wavelenght independent 3/4(1+cos(th)**2)
    improves the results for back and forward scatterring by ~1.5%.

    Anthony Bucholtz, "Rayleigh-scattering calculations for the terrestrial atmosphere", 
    Applied Optics 34, no. 15 (May 20, 1995): 2765-2773.  

    R. B Miles, W. R Lempert, and J. N Forkey, "Laser Rayleigh scattering", 
    Measurement Science and Technology 12 (2001): R33-R51.
    """
    p_e = rh_to_pressure(rh, temperature)
    r = rho_atmosphere(wavelength, C, p_e, pressure)
    gamma = r / (2 - r)
    f1 = 3 / (4 * (1 + 2 * gamma))
    f2 = (1 + 3 * gamma) + (1 - gamma) * (np.cos(theta)) ** 2
    p = f1 * f2
    return p


def physical_constants() -> dict:
    """Physical constants.

    Returns
    -------
    dict
        h : float
            ...
        c : float
            ...
        k_b : float
            ...
        R : float
            ...
    
    Notes
    -----
    .. TODO:: Finish docstring.
    """
    return {
        'h': 6.626070040e-34,
        'c': 299792458.,
        'k_b': 1.38064852 * 10 ** -23,
        'R': 8.314510
    }


def pressure_to_rh(partial_pressure:float|np.ndarray, temperature:float|np.ndarray) -> float|np.ndarray:
    """Convert water vapour partial pressure to relative humidity.
    
    Parameters
    ----------
    partial_pressure : float or ndarray
        Water vapour partial pressure [hPa].
    temperature : float or ndarray
        Temperature [K].
    
    Returns
    -------
    rh : float or ndarray
        Relative humidity from 0 to 100 [%]
    """
    svp = saturation_vapor_pressure(temperature)
    rh = partial_pressure / svp * 100
    return rh


def rh_to_pressure(rh:float|np.ndarray, temperature:float|np.ndarray) -> float|np.ndarray:
    """Convert relative humidity to water vapour partial pressure.
    
    Parameters
    ----------
    rh : float or ndarray
        Relative humidity from 0 to 100 [%].
    temperature : float or ndarray
        Temperature [K].
    
    Returns
    -------
    p_wv : float or ndarray
        Water vapour pressure [hPa].
    """
    svp = saturation_vapor_pressure(temperature)
    h = rh / 100
    p_wv = h * svp
    return p_wv


def rho_atmosphere(wavelength:float, C:float, p_e:float|np.ndarray,
                   p_t:float|np.ndarray) -> float|np.ndarray:
    """Calculate the depolarization factor of the atmosphere.
    
    Parameters
    ----------
    wavelength : float
        Wavelength [nm].
    C : float
        CO2 concentration [ppmv].
    p_e : float or ndarray
        water-vapor pressure [hPa].
    p_t : float or ndarray
        total air pressure [hPa].
    
    Returns
    -------
    rho : float or ndarray
        Depolarization factor.
    """
    F_k = kings_factor_atmosphere(wavelength, C, p_e, p_t)
    rho = (6 * F_k - 6) / (7 * F_k + 3)
    return rho


def saturation_vapor_pressure(temperature:float):
    """Saturation vapor pressure of water of moist air.
    
    Parameters
    ----------
    temperature : float
        Atmospheric temperature [K].
    
    Returns
    -------
    E : float
        Saturation vapor pressure [hPa].
    """
    T = temperature
    E = np.exp(1.2378847e-5 * T**2 - 1.9121316e-2 * T + 33.93711047 - 6343.1645 / T) / 100
    return E


def sigma_rayleigh(wavelength:float, pressure:float|np.ndarray, temperature:float|np.ndarray, 
                   C:float, rh:float|np.ndarray) -> float|np.ndarray:
    """Calculates the Rayleigh-scattering cross section per molecule.
    
    Parameters
    ----------
    wavelength : float
        Wavelength [nm].
    pressure : float or ndarray
        The atmospheric pressure [hPa].
    temperature : float or ndarray
        The atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    rh : float or ndarray
        Relative humidity from 0 to 100 [%].
    
    Returns
    -------
    sig : float or ndarray
        Rayleigh-scattering cross section [m2].
    """

    p_e = rh_to_pressure(rh, temperature)

    # Calculate properties of standard air
    n = air_refractive_index(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ASSUME_AIR_IDEAL)

    # Wavelength of radiation
    wl_m = wavelength  # nm

    # King's correction factor
    f_k = kings_factor_atmosphere(wavelength, C, p_e, pressure)  # no units

    # first part of the equation
    f1 = (24 * np.pi**3) / (wl_m**4 * (N*1e-18)**2)
    # second part of the equation
    f2 = (n**2 - 1)**2 / (n**2 + 2)**2

    sig = f1 * f2 * f_k
    return sig


def rayleigh_scattering(wavelength:float, pressure:float|np.ndarray, temperature:float|np.ndarray, 
                        C:float, rh:float|np.ndarray) -> tuple:
    """Calculate the molecular volume backscatter coefficient and extinction coefficient.

    Parameters
    ----------
    wavelength : float
        Wavelength in nanometers [nm].
    pressure : float or ndarray
        Atmospheric pressure [hPa].
    temperature : float or ndarray
        Atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    rh : float or ndarray
        Relative humidity as a percentage (0 to 100).

    Returns
    -------
    beta_mol : float or ndarray
        Molecular backscatter coefficient [m^{-1}*sr^{-1}].
    alpha_mol : float or ndarray
        Molecular extinction coefficient [m^{-1}].

    References
    ----------
    Bucholtz, A.: Rayleigh-scattering calculations for the terrestrial atmosphere, 
    Appl. Opt. 34, 2765-2773 (1995).
    A. Behrendt and T. Nakamura, "Calculation of the calibration constant of polarization lidar and 
    its dependency on atmospheric temperature," Opt. Express, vol. 10, no. 16, pp. 805-817, 2002.

    Notes
    -----
    - Based on the Python source code of Ioannis Binietoglou's [repo](https://bitbucket.org/iannis_b/lidar_molecular).

    **History**

    - 2017-12-16: First edition by Zhenping.
    - 2024-12-03: AI-Translated.
    """

    beta_mol = beta_pi_rayleigh(wavelength, pressure, temperature, C, rh)
    alpha_mol = alpha_rayleigh(wavelength, pressure, temperature, C, rh)
    return beta_mol, alpha_mol


def calc_profiles(met_profiles:list, wavelengths:list=[355, 387, 407, 532, 607, 1058, 1064],
                  CO2:float=400, flagPicassoComparison:bool=False) -> dict:
    """For a list of xarray averaged meteorology profiles, calculate the rayleigh scattering.

    Parameters
    ----------
    met_profiles : list
        List of xarray averaged meteorology profiles
    wavelengths : list, optional
        List of wavelengths to perform the rayleigh scattering calculation [nm].
        Default is [355, 387, 407, 532, 607, 1058, 1064]
    CO2 : float, optional
        CO2 concentration [ppmv].
    flagPicassoComparison : bool, optional
        If true, use Picasso values and logic. Default is False.
   
    Returns
    -------
    dict
        Molecular backscatter, extinction, and number density profiles for each
        input wavelength.
    """

    print('len mean_profiles', len(met_profiles))
    shp = (len(met_profiles), met_profiles[0].height.shape[0])
    print('shape of the molecular scattering', shp)
    print('for the wavelengths ', wavelengths)
    m_p = defaultdict(lambda: np.zeros(shp))

    for i, p in enumerate(met_profiles):
        if flagPicassoComparison:
            CO2 = 380
            p['rh'].values = np.ones(shp[1])*0.7

        for wv in wavelengths:
            m_p[f'mBsc_{wv}'][i, :], m_p[f'mExt_{wv}'][i, :] = rayleigh_scattering(
                wavelength=wv,
                pressure=p['pressure'].values/100,
                temperature=p['temperature'].values,
                C=CO2,
                rh=p['rh'].values*100
            )
        m_p['number_density'][i, :] = number_density_at_pt(
            pressure=p['pressure'].values/100, 
            temperature=p['temperature'].values,
            relative_humidity=p['rh'].values*100,
            ideal=True
        )

    return dict(m_p)


def calc_2d(met:xr.Dataset, wavelengths:list=[355, 387, 407, 532, 607, 1058, 1064], 
            CO2:float=400, flagPicassoComparison:bool=False) -> xr.Dataset:
    """For a two dimensional xarray dataset of meteorology profiles, calculate the rayleigh scattering.
   
    Parameters
    ----------
    met : Dataset
        Dataset of meteorology profiles (temperature, pressure, relative humidity, specific humidity).
    wavelengths : list, optional
        List of wavelengths to perform the rayleigh scattering calculation [nm].
        Default is [355, 387, 407, 532, 607, 1058, 1064]
    CO2 : float, optional
        CO2 concentration [ppmv].
    flagPicassoComparison : bool, optional
        If true, use Picasso values and logic. Default is False.
   
    Returns
    -------
    ds_mol : xr.Dataset
        xarray dataset with dimensions time and height and variables molecular
        backscatter (mBsc) and molecular extinction (mExt) per wavelength.
    """

    ds_mol = xr.Dataset(
        coords=dict(
            time=met['time'],
            height=met['height'],
        )
    )

    if flagPicassoComparison:
        CO2 = 380
        met['rh'].values = np.ones_like(met['rh'])*0.7
    
    for wv in wavelengths:
        mBsc, mExt = rayleigh_scattering(
            wv, met['pressure'].values/100, met['temperature'].values, CO2, met['rh'].values*100)

        ds_mol[f'mBsc_{wv}'] = (['time', 'height'], mBsc)
        ds_mol[f'mExt_{wv}'] = (['time', 'height'], mExt)
        
    return ds_mol

