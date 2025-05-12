
**************
Preprocessing
**************

Setting first range gate
=========================

The first range gate has to be set correctly in terms of ``first_range_gate_indx`` and ``first_range_gate_height``.
Each of these quantities has to be set for each channel (mostly depending on the data acquisition).

.. note::
    In the matlab version only the index could be set channel-depended, i.e., only one one height could be set for **all** channels.
    A pragmatic fix is introduced below, please be aware of the implications for high precision applications.

.. code-block:: python

    data_cube.polly_config_dict['first_range_gate_indx']
    >>> [252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252]
    data_cube.polly_config_dict['first_range_gate_height']
    >>> [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 6.75, 6.75, 6.75, 6.75, 6.75]

To check the correct setting of the bin height, the ``preproSignal`` and the ``sigBGCor`` can be compared close to the trigger

.. code-block:: python

    # to keep the preproSignal, the collect_debug has to be set
    # data_cube.preprocessing(collect_debug=True)

    height = data_cube.retrievals_highres['height']
    i = 0
    sig = np.squeeze(
        data_cube.retrievals_highres['preproSignal'][i,:,data_cube.gf(532,'total', 'FR')])
    sigNR = np.squeeze(
        data_cube.retrievals_highres['preproSignal'][i,:,data_cube.gf(532,'total', 'NR')])
    bin_first_range_gate = np.array(
        data_cube.polly_config_dict['first_range_gate_indx'])[data_cube.gf(532,'total', 'FR')]
    bin_first_range_gateNR = np.array(
        data_cube.polly_config_dict['first_range_gate_indx'])[data_cube.gf(532,'total', 'NR')]
    
    fig, ax = plt.subplots(1, figsize=(6, 3), sharey=True)
    ax.plot(sig, label='532_total_FR')
    ax.plot(sigNR, label='532_total_NR')
    ax.axvline(bin_first_range_gate, c='C0', ls="-.")
    ax.axvline(bin_first_range_gateNR, c='C1', ls=":")
    ax.set_xlim(240,270)
    ax.legend()
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax.set_xlabel('Bin'); ax.set_ylabel('preproSignal [a.u.]')

    sig = np.squeeze(
        data_cube.retrievals_highres['sigBGCor'][i,:,data_cube.gf(532,'total', 'FR')])
    sigNR = np.squeeze(
        data_cube.retrievals_highres['sigBGCor'][i,:,data_cube.gf(532,'total', 'NR')])

    fig, ax = plt.subplots(1, figsize=(6, 3), sharey=True)
    ax.plot(height, sig, label='532_total_FR')
    ax.plot(height, sigNR, label='532_total_FN')
    ax.set_xlim(-5,95)
    ax.legend()
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.set_xlabel('Height [m]'); ax.set_ylabel('sigBGCor [a.u.]')

Where the ``preproSignal`` (left) still contains the pretrigger and the laser pulse, whereas the ``sigBGCor`` (right) only starts with the bin,
that contains a 0 count rate.

.. image:: img/first_range_gate_indx.png
  :width: 100%


The ``first_range_gate_height`` affects the range corrected signal as well as the mapping from time to range. In the original labview
analysis program, this was solved by having distinct range arrays for FR and NR, which turned out to be not extremely practical.
In the matlab version of the processing chain the problem was not addressed.
A pragmatic fix that is currently implemented in :func:`ppcpy.preprocess.pollyPreprocess.pollyPreprocess` fixes the effect on the range corrected
signal, but ignores the effect of potentially shifting the range array.
As long as the difference between the first range gate heights of different channels (or data acquisitions) is less than the range resolution, 
this shift can be ignored. Additionally, almost all subsequent products are smoothed, which further smears out the effect.

.. code:: python

    data_dict['range'] = np.arange(0, sigBGCor.shape[1]) * hRes + firstBinHeight[0]

    correction_firstBinHight = ((
        (np.arange(0, sigBGCor.shape[1]) * hRes)[:,np.newaxis] + firstBinHeight)**2
        / data_dict['range'][:,np.newaxis]**2)
    data_dict['sigBGCor'] = data_dict['sigBGCor'] * correction_firstBinHight[np.newaxis,:,:]

Later, when the range corrected signal is calculated, the the count rates are converted with the 'correct' range.

.. code:: python

    data_cube.polly_config_dict['first_range_gate_height'] = [
        3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    data_cube.preprocessing(collect_debug=False)
    sig = np.squeeze(data_cube.retrievals_highres['RCS'][0,:,data_cube.gf(532,'total', 'FR')].copy())
    sigNR = np.squeeze(data_cube.retrievals_highres['RCS'][0,:,data_cube.gf(532,'total', 'NR')].copy())
    
    data_cube.polly_config_dict['first_range_gate_height'] = [
        6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    data_cube.preprocessing(collect_debug=False)
    sig_all6m = np.squeeze(data_cube.retrievals_highres['RCS'][0,:,data_cube.gf(532,'total', 'FR')].copy())
    sigNR_all6m = np.squeeze(data_cube.retrievals_highres['RCS'][0,:,data_cube.gf(532,'total', 'NR')].copy())
    
    data_cube.polly_config_dict['first_range_gate_height'] = [
        3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75]
    data_cube.preprocessing(collect_debug=False)
    sig_all3m = np.squeeze(data_cube.retrievals_highres['RCS'][0,:,data_cube.gf(532,'total', 'FR')].copy())
    sigNR_all3m = np.squeeze(data_cube.retrievals_highres['RCS'][0,:,data_cube.gf(532,'total', 'NR')].copy())



.. image:: img/first_range_gate_height_correction.png
  :width: 100%


