
**************
Preprocessing
**************



Setting first range gate
=========================

The first range gate has to be set correctly in terms of ``first_range_gate_indx`` and ``first_range_gate_height``.
Each of these quantities has to be set for each channel (mostly depending on the data acquisition).

.. note::
    In the matlab version only the index could be set channel-depended.

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


.. image:: img/first_range_gate_indx.png
  :width: 100%

