import numpy as np

def physical_constants():
    """
    Basic physics constants used in calculations.
    Values taken from NIST (2014 values).
    """
    return {
        'h': 6.626070040e-34,  # Planck constant [J·s]
        'c': 299792458.0,  # Speed of light [m/s]
        'k_b': 1.38064852e-23,  # Boltzmann constant [J/K]
        'R': 8.314510,  # Molar gas constant [J/(mol·K)]
    }

def molar_mass_dry_air(C):
    """
    Molar mass of dry air as a function of CO2 concentration.
    C: CO2 concentration [ppmv]
    Returns: Molar mass of dry air [kg/mol]
    """
    C1 = 400.0
    return 1e-3 * (28.9635 + 12.011e-6 * (C - C1))

def compressibility_of_moist_air(pressure, temperature, molar_fraction):
    """
    Compressibility of moist air.
    pressure: Total pressure [hPa]
    temperature: Temperature [K]
    molar_fraction: Molar fraction of water vapor
    """
    p = pressure * 100.0  # Convert hPa to Pa
    T = temperature
    Tc = T - 273.15  # Convert K to °C
    
    a0, a1, a2 = 1.58123e-6, -2.9331e-8, 1.1043e-10
    b0, b1 = 5.707e-6, -2.051e-8
    c0, c1 = 1.9898e-4, -2.376e-6
    d0, d1 = 1.83e-11, -7.65e-9
    
    Xw = molar_fraction
    
    Z = (1 - (p / T) * 
         (a0 + a1 * Tc + a2 * Tc ** 2 + (b0 + b1 * Tc) * Xw +
          (c0 + c1 * Tc) * Xw ** 2) + 
         (p / T) ** 2 * (d0 + d1 * Xw ** 2))
    
    return Z

def moist_air_density(pressure, temperature, C, Xw):
    """
    Calculate the density of moist air.
    pressure: Total pressure [hPa]
    temperature: Temperature [K]
    C: CO2 concentration [ppmv]
    Xw: Molar fraction of water vapor
    """
    const = physical_constants()
    Ma = molar_mass_dry_air(C)  # Dry air molar mass [kg/mol]
    Mw = 0.018015  # Water vapor molar mass [kg/mol]
    
    Z = compressibility_of_moist_air(pressure, temperature, Xw)
    
    P = pressure * 100.0  # Convert hPa to Pa
    T = temperature
    
    rho = (P * Ma / (Z * const['R'] * T)) * (1 - Xw * (1 - Mw / Ma))
    rho_air = (1 - Xw) * P * Ma / (Z * const['R'] * T)
    rho_wv = Xw * P * Mw / (Z * const['R'] * T)
    
    return rho, rho_air, rho_wv

def molar_fraction_water_vapor(pressure, temperature, relative_humidity):
    """
    Molar fraction of water vapor.
    pressure: Total pressure [hPa]
    temperature: Temperature [K]
    relative_humidity: Relative humidity [%] (0-100)
    """
    f = enhancement_factor_f(pressure, temperature)
    svp = saturation_vapor_pressure(temperature)
    
    p_wv = (relative_humidity / 100.0) * f * svp
    Xw = p_wv / pressure
    return Xw

def enhancement_factor_f(pressure, temperature):
    """
    Enhancement factor for moist air.
    pressure: Total pressure [hPa]
    temperature: Temperature [K]
    """
    T = temperature
    p = pressure * 100.0  # Convert hPa to Pa
    return 1.00062 + 3.14e-8 * p + 5.6e-7 * (T - 273.15) ** 2

def saturation_vapor_pressure(temperature):
    """
    Calculate saturation vapor pressure based on temperature.
    temperature: Temperature [K]
    Returns: Saturation vapor pressure [hPa]
    """
    T = temperature
    T_c = T - 273.15  # Convert K to °C
    return 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))

def air_refractive_index(wavelength, pressure, temperature, C, relative_humidity):
    """
    Calculate the refractive index of air.
    wavelength: Wavelength [nm]
    pressure: Atmospheric pressure [hPa]
    temperature: Atmospheric temperature [K]
    C: CO2 concentration [ppmv]
    relative_humidity: Relative humidity [%]
    Returns: Refractive index of air.
    """
    Xw = molar_fraction_water_vapor(pressure, temperature, relative_humidity)
    rho_axs = moist_air_density(1013.25, 288.15, C, 0)[0]
    rho_ws = moist_air_density(13.33, 293.15, 0, 1)[0]
    
    _, rho_a, rho_w = moist_air_density(pressure, temperature, C, Xw)
    
    n_axs = n_standard_air_with_CO2(wavelength, C)
    n_ws = n_water_vapor(wavelength)
    
    n_air = 1 + (rho_a / rho_axs) * (n_axs - 1) + (rho_w / rho_ws) * (n_ws - 1)
    return n_air

# Additional helper functions such as `n_standard_air_with_CO2`, `n_water_vapor`, 
# `n_standard_air`, etc., should be similarly converted.

def sigma_pi_cabannes(wavelength, pressure, temperature, C, rh):
    """
    Calculate the backscattering cross section for the Cabannes line.

    Parameters:
    ----------
    wavelength : float
        Light wavelength in nanometers [nm].
    pressure : float
        Atmospheric pressure [hPa].
    temperature : float
        Atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    rh : float
        Relative humidity as a percentage (0 to 100).

    Returns:
    -------
    sigma_pi : float
        Backscattering cross section of the Cabannes line [m^2*sr^{-1}].
    """

    ASSUME_AIR_IDEAL = True

    p_e = rh_to_pressure(rh, temperature)
    assert False, "epsilon_atmosphere and n_air are not given as functions"
    epsilon = epsilon_atmosphere(wavelength, C, p_e, pressure)
    n = n_air(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ASSUME_AIR_IDEAL)
    
    # Convert wavelength to meters
    lamda_m = wavelength * 1e-9

    # Break into three factors for clarity
    f1 = 9 * np.pi**2 / (lamda_m**4 * N**2)
    f2 = ((n**2 - 1)**2) / ((n**2 + 2)**2)
    f3 = 1 + 7 / 180 * epsilon

    sigma_pi = f1 * f2 * f3
    return sigma_pi


def beta_pi_cabannes(wavelength, pressure, temperature, C, rh):
    """
    Calculate the backscattering coefficient for the Cabannes line.

    Parameters:
    ----------
    wavelength : float
        Light wavelength in nanometers [nm].
    pressure : float
        Atmospheric pressure [hPa].
    temperature : float
        Atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    rh : float
        Relative humidity as a percentage (0 to 100).

    Returns:
    -------
    beta_pi : float
        Backscattering coefficient of the Cabannes line [m^{-1}*sr^{-1}].
    """
    
    ASSUME_AIR_IDEAL = True

    sigma_pi = sigma_pi_cabannes(wavelength, pressure, temperature, C, rh)
    N = number_density_at_pt(pressure, temperature, rh, ASSUME_AIR_IDEAL)

    beta_pi = N * sigma_pi
    return beta_pi



def pressure_to_rh(partial_pressure, temperature):
    """
    Convert water vapor partial pressure to relative humidity.

    Parameters:
    ----------
    partial_pressure : float
        Water vapor partial pressure [hPa].
    temperature : float
        Temperature in Kelvin [K].

    Returns:
    -------
    rh : float
        Relative humidity as a percentage (0 to 100).
    """
    
    svp = saturation_vapor_pressure(temperature)
    rh = (partial_pressure / svp) * 100
    return rh




def number_density_at_pt(pressure, temperature):
    """
    Calculate the number density of gas at a given pressure and temperature.

    Parameters:
        pressure (float): Pressure [hPa].
        temperature (float): Temperature [K].

    Returns:
        float: Number density of gas [molecules/cm³].

    Notes:
        Number density is calculated using the ideal gas law:
            n = P / (k_B * T),
        where k_B is Boltzmann's constant, and n is expressed in molecules/cm³.
    """
    const = physical_constants()
    P = pressure * 100.0  # Convert from hPa to Pa.
    T = temperature  # Temperature in Kelvin.
    # Number density in molecules/cm³
    return (P / (const['kB'] * T)) * 1e-6


def n_standard_air_with_CO2(wavelength, C):
    """
    Calculate the refractive index of air at a specific wavelength with CO2 concentration.

    Parameters:
        wavelength (float): Wavelength [nm].
        C (float): CO2 concentration [ppmv].

    Returns:
        float: Refractive index of air for the given CO2 concentration.
    """
    C2 = 450.0
    n_as = n_standard_air(wavelength)
    return 1 + (n_as - 1) * (1 + 0.534e-6 * (C - C2))


def n_standard_air(wavelength):
    """
    Calculate the refractive index of standard air at a given wavelength.

    Parameters:
        wavelength (float): Wavelength [nm].

    Returns:
        float: Refractive index of standard air.
    """
    wl_um = wavelength / 1000.0  # Convert nm to µm.
    s = 1 / wl_um
    c1, c2, c3, c4 = 5792105.0, 238.0185, 167917.0, 57.362
    return 1 + (c1 / (c2 - s**2) + c3 / (c4 - s**2)) * 1e-8


def n_water_vapor(wavelength):
    """
    Calculate the refractive index of water vapor.

    Parameters:
        wavelength (float): Wavelength [nm].

    Returns:
        float: Refractive index of water vapor.
    """
    wl_um = wavelength / 1000.0  # Convert nm to µm.
    s = 1 / wl_um
    c1, c2, c3, c4, c5 = 1.022, 295.235, 2.6422, 0.032380, 0.004028
    return 1 + c1 * (c2 + c3 * s**2 - c4 * s**4 + c5 * s**6) * 1e-8


def saturation_vapor_pressure(temperature):
    """
    Calculate the saturation vapor pressure of water as a function of temperature.

    Parameters:
        temperature (float): Temperature [K].

    Returns:
        float: Saturation vapor pressure [hPa].

    Notes:
        Uses the Tetens equation for vapor pressure:
            es = 6.112 * exp((17.62 * Tc) / (243.12 + Tc)),
        where Tc = temperature in Celsius. The result is in hPa.
    """
    Tc = temperature - 273.15  # Convert temperature from Kelvin to Celsius.
    return 6.112 * np.exp((17.62 * Tc) / (243.12 + Tc))


def rh_to_pressure(rh, temperature):
    """
    Calculate the partial pressure of water vapor given relative humidity and temperature.

    Parameters:
        rh (float): Relative humidity [%] (0-100 scale).
        temperature (float): Temperature [K].

    Returns:
        float: Partial pressure of water vapor [hPa].

    Notes:
        Partial pressure is calculated using the equation:
            e = (rh / 100) * es,
        where es is the saturation vapor pressure, calculated based on temperature.
    """
    es = saturation_vapor_pressure(temperature)  # Saturation vapor pressure [hPa].
    return (rh / 100.0) * es





def rayleigh_scattering(wavelength, pressure, temperature, C, rh):
    """
    Calculate the molecular volume backscatter coefficient and extinction coefficient.

    Parameters:
    ----------
    wavelength : float
        Wavelength in nanometers [nm].
    pressure : float
        Atmospheric pressure [hPa].
    temperature : float
        Atmospheric temperature [K].
    C : float
        CO2 concentration [ppmv].
    rh : float
        Relative humidity as a percentage (0 to 100).

    Returns:
    -------
    beta_mol : float
        Molecular backscatter coefficient [m^{-1}*sr^{-1}].
    alpha_mol : float
        Molecular extinction coefficient [m^{-1}].

    References:
    ----------
    Bucholtz, A.: Rayleigh-scattering calculations for the terrestrial atmosphere, 
    Appl. Opt. 34, 2765-2773 (1995).
    A. Behrendt and T. Nakamura, "Calculation of the calibration constant of polarization lidar and 
    its dependency on atmospheric temperature," Opt. Express, vol. 10, no. 16, pp. 805-817, 2002.

    History:
    -------
    First edition by Zhenping, 2017-12-16. 
    Based on the Python source code of Ioannis Binietoglou's 
    [repo](https://bitbucket.org/iannis_b/lidar_molecular).
    """

    beta_mol = beta_pi_cabannes(wavelength, pressure, temperature, C, rh)
    alpha_mol = alpha_rayleigh(wavelength, pressure, temperature, C, rh)
    
    return beta_mol, alpha_mol




