
********************
Conventions
********************

Structure of data_cube
=========================

The ``data_cube`` (:py:meth:`~ppcpy.interface.picassoProc.PicassoProc`) provides all the methods to do the processing as well as the data produces.
Most importantly there are two variables for the averaged profiles ``retrievals_profile`` and for the high resolution time-height data ``retrievals_highres``.
Both contain dictionaries with the content outlined below. 


data_cube.retrievals_profile
-----------------------------

Each key with retrieved profiles hold a list with all the ``data_cube.clFreeGrps`` containing subdictionaries with individual channels.
An list of available retrievals is kept in

.. code:: python

    data_cube.retrievals_profile['avail_optical_profiles']
    >>> ['klett', 'raman', 'klett_OC', 'raman_OC']

additionally the inputs for the retrievals (aggregated profiles) are also stored, e.g,

.. code:: python

    ['RCS', 'sigBGCor', 'BG', 'refH', 'sigTCor', 'BGTCor', 'overlap', 'sigOLCor', 'BGOLCor']


data_cube.retrievals_highres
-----------------------------

Holds all the hight temporal-resolution data, including auxiliary variables like ``range``, ``height``, ``time``, ``time64``.

These are the inputs at various levels of processing:

.. code:: python

    ['preproSignal', 'BG', 'sigBGCor', 'SNR', 'RCS', 'sigTCor', ...]

flags, such as:

.. code:: python

    ['lowSNRMask', 'shutterOnMask', 'fogMask', 'mask607Off', 'mask387Off', 'mask407Off', 'mask1064_RROff', ...]


and geophysical retrievals:

.. code:: python

    ['attBsc_532_total_FR', 'voldepol_355_total_FR', 'quasiBscV1_1064_total_FR', 'tcMaskV2']

A full list is available using ``data_cube.retrievals_highres.keys()``
The exact content depends on the usage of the ``collect_debug`` flag.


Channel naming
=========================

Extensive discussion in `Issue 303 of the matlab processing chain <https://github.com/PollyNET/Pollynet_Processing_Chain/issues/303>`_
came to the conclusion to work with

.. code::

    {wavelength}_{total|cross|parallel|rotraman}_{NR|FR|DFOV}

For example

* ``355_total_FR``
* ``532_cross_NR``
* ``1064_total_FR``
* ``532_cross_DFOV``
* ...