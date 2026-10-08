# PicassoPy: PollyNET Processing Chain python version

[![GitHub License](https://img.shields.io/badge/License-GPLv3-green.svg)](https://github.com/PollyNET/PicassoPy/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://pollynet.github.io/PicassoPy-doc/html/index.html)
[![GitHub Tag](https://img.shields.io/github/v/tag/PollyNET/PicassoPy?label=latest&color=blue&logo=github)](https://github.com/PollyNET/PicassoPy/tags)
[![GitHub commits since latest](https://img.shields.io/github/commits-since/PollyNET/PicassoPy/latest.svg?color=blue)](https://github.com/PollyNET/PicassoPy/commits/main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16813294.svg)](https://doi.org/10.5281/zenodo.23242759)

This repository contains the python version of the PollyNET Processing Chain ([Klamt et al, 2024](https://doi.org/10.5281/zenodo.13379737)) responsible for the automatic processing and visualization of lidar data from [PollyNET](https://polly.tropos.de/) ([Baars et al, 2016](https://doi.org/10.5194/acp-16-5111-2016)). PollyNET is an independent, voluntary, and scientific network initiated by the Leibniz Institute for Tropospheric Research (TROPOS). The network consists of novel multiwavelength Raman polarization lidars of type PollyXT ([Engelmann et al, 2016](https://doi.org/10.5194/amt-9-1767-2016), [2025](https://www.igf.fuw.edu.pl/m/conferences/34/a7/34a7d6cf-733c-4ab7-a532-fb3d9991561a/BookELC2025.pdf)) deployed by TROPOS and several partners at locations around the world, like Dushanbe Tajikistan, Mindelo Cabo Verde, and Limassol Cyprus among others. They can measure vertical profiles of tiny, airborne particles (aerosol and clouds) and water vapor with high temporal and vertical resolution in the atmosphere. All systems are designed for continuous, unattended operation. For the most advanced version, the system has more than 12 channels, including 8 far-range channels and 4 near-range channels. It can deliver vertical profiles of the particle backscatter coefficient, the particle extinction coefficient, the particle lidar ratio, the particle linear depolarization ratio and respective Ångström exponents but also water vapor mixing ratio profile at night-time.

The program in this repository is the standard to retrieve products from PollyNET measurements. It automatically calibrates the lidar data, e.g., with respect to depolarization and water vapor channels, and applies several corrections (e.g., dead time, background, etc.). It retrieves vertical profiles of optical properties aggregated over cloud-free periods. Next to these standard products, several quicklooks (time-height cross sections) including quasi-products and a target categorization ([Baars et al, 2017](https://doi.org/10.5194/amt-10-3175-2017)) are provided.

Further information about the program and how to use it can be found in the [documentation](https://pollynet.github.io/PicassoPy-doc/html/index.html).

## Installation

```
# setup a virtual environment (either conda or venv)
git clone https://github.com/PollyNET/PicassoPy.git

cd PicassoPy
python -m pip install -e .
```

## Usage

```
python3 tests\picassopy_testing.py --picasso_config_file "pollynet_processing_chain_config.json" \
    --date 20240308 --device pollyxt_cpv \
    --level0_file_to_process "2024_03_08_Fri_CPV_00_00_01.nc"
```

## Contribute

Your contribution is welcomed! Just have a look into the issue, some of them are quite easy to fix or file one yourself detailing a bug you found.
Additions to the documentation or tests is also always encouraged. Also, if you want to contribute more sophisticated algorithms feel free to file a pull request.

## History

This program is based on the foundation the [PollyNET Processing Chaine](https://github.com/PollyNET/Pollynet_Processing_Chain).

## License

This program is licensed under the GNU GPL-3.0 license (see [LICENSE](https://github.com/PollyNET/PicassoPy/blob/f3a6d7fa70ed4c0efa50c8a68346658fb0998c2f/LICENSE)).

## References

* Andi Klamt et al., PollyNET/Pollynet_Processing_Chain: Version 4.0, 27 August 2024, [https://doi.org/10.5281/zenodo.13379737](https://doi.org/10.5281/zenodo.13379737).
* Ronny Engelmann et al., 'The Automated Multiwavelength Raman Polarization and Water-Vapor Lidar PollyXT: The neXT Generation', Atmospheric Measurement Techniques 9, no. 4 (2016): 1767–84, [https://doi.org/10.5194/amt-9-1767-2016](https://doi.org/10.5194/amt-9-1767-2016).
* Holger Baars et al., 'An Overview of the First Decade of PollyNET: An Emerging Network of Automated Raman-Polarization Lidars for Continuous Aerosol Profiling', Atmospheric Chemistry and Physics 16, no. 8 (2016): 5111–37, [https://doi.org/10.5194/acp-16-5111-2016](https://doi.org/10.5194/acp-16-5111-2016).
* Holger Baars et al., 'Target Categorization of Aerosol and Clouds by Continuous Multiwavelength-Polarization Lidar Measurements', Atmospheric Measurement Techniques 10, no. 9 (2017): 3175–201, [https://doi.org/10.5194/amt-10-3175-2017](https://doi.org/10.5194/amt-10-3175-2017).
* Guangyao Dai et al., 'Calibration of Raman Lidar Water Vapor Profiles by Means of AERONET Photometer Observations and GDAS Meteorological Data', Atmospheric Measurement Techniques 11, no. 5 (2018): 2735–48, [https://doi.org/10.5194/amt-11-2735-2018](https://doi.org/10.5194/amt-11-2735-2018).
