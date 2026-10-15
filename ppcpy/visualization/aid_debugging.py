
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_cals(d:dict, param:str, used:dict=None):
    """Plot the calibration constants.

    Produces a scatter plot of the stored calibration constants (CC).
    CCs retrieved form the measurement are marked as filled circles,
    while those loaded from the database are marked as hollow circles.
    The chosen optimal CC and its default value are marked by horizontal
    gray dotted and dashed lines respectively.

    Parameters
    ----------
    d : dict
        Dict storing all CCs. ``pol_cali`` or ``LC``.
    param : str
        Name of the parameter to extract. 
    used : dict, optional
        Dict storing the used CCs. LCused or etaused.
        Will produce a horizontal dashed line in the plot if added.
        Default is None.
    
    Returns
    -------
    fig : figure
        Matplotlib figure object.
    ax : axis
        Matplotlib axis object.

    Examples
    --------
    
    >>> plot_cals(data_cube.pol_cali, 'eta', used=data_cube.etaused)
    >>> plot_cals(data_cube.LC, 'LC', used=data_cube.LCused)
    
    """

    channels = set()
    for k, v in d.items():
        channels.update(v.keys())
    print('all channels', channels)

    fig, ax = plt.subplots(len(channels), figsize=(8, len(channels)*2), sharex=True)
    for i, c in enumerate(channels):
        guess_yscale = []

        if used and c in used.keys():
            ax[i].axhline(used[c][param], color='dimgrey', ls=':', label="used")
                    
        for k, v in d.items():
            if not c in v:
                continue

            if "default" in k:
                ax[i].axhline(d[k][c][0][param], color='dimgray', alpha=0.6, ls='--', label="default")
                guess_yscale.append(d[k][c][0][param]*0.95)
                guess_yscale.append(d[k][c][0][param]*1.05)
                continue
            
            if 'db' in k:
                marker = 'o'
                fillstyle = 'none'
            else:
                marker = '.'
                fillstyle = 'full'             

            time_mean = np.array([np.mean([e['time_start'], e['time_end']]) for e in v[c]]).astype('datetime64[s]')
            eta = [e[param] for e in v[c]]
            # if used and c in used.keys():
            #     ax.axhline(used[c][param], color='dimgrey', ls=':', label="used")
            
            ax[i].plot(time_mean, eta, marker, label=k, fillstyle=fillstyle)
            guess_yscale.append(np.min(eta)*0.95)
            guess_yscale.append(np.max(eta)*1.05)
            guess_yscale.append(np.mean(eta)*1.2)
            guess_yscale.append(np.mean(eta)*0.8)

        if guess_yscale:
            ax[i].set_ylim(np.max([0, np.min(guess_yscale)]), np.max(guess_yscale))
        ax[i].set_ylabel(param)
        ax[i].legend()
        ax[i].set_title(c)
        ax[i].xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
        ax[i].xaxis.set_minor_locator(matplotlib.dates.HourLocator(interval=1))
        ax[i].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M\n%d.%m.'))
        
    return fig, ax

    
    
    
def plot_outerFOV_debug(debug, height):
    """diagnostics plot for the outer field of view depol calibration via transfer
    
    
    
    Examples
    ---------
    
    >>> debug = data_cube.polarizationCaliOuterFOV(store_debug=True, collect_vars_plot=True)
    >>> import ppcpy.visualization.aid_debugging as ad
    >>> ad.plot_outerFOV_debug(debug, np.array(data_cube.retrievals_highres['height'])/1000)
    """


    #height = np.array(data_cube.retrievals_highres['height'])/1000

    figs_out = []

    for o in debug:
        #print(o)
        cldFreeTime64 = o['cldFreeTime64']
        vdr_inner_min = o['vdr_inner_min']
        height_mask = o['height_mask']
        eta_outer_guess = o['eta_outer_guess']
        vdr_inner = o['vdr_inner']
        eta_outer = o['eta_outer']
        vdr_outer = o['vdr_outer']
        vdr_outer_dash = o['vdr_outer_dash']
        Hr = o['Hr']
        Hr_outer_init = o['Hr_outer_init']
        Hr_fixed = o['Hr_fixed']
        eta_outer_opt = o['eta']
        rmse = o['rmse']
        median_ratio = o['median_ratio']
        dt_grp = cldFreeTime64[0].astype('datetime64[us]').astype('O')

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from scipy.ndimage import gaussian_filter1d

        s_title = (
            f"{cldFreeTime64[0].astype('datetime64[us]').astype('O'):%Y-%m-%d %H%M%S}-"
            f"{cldFreeTime64[1].astype('datetime64[us]').astype('O'):%H%M%S}"
            f"   Heights {height[height_mask][0]:.1f} - "
            f"{height[height_mask][-1]:.1f} km"
        )

        # Figure layout:
        #   [ height profile | eta scatter ]
        #   [                | vdr scatter ]
        fig = plt.figure(figsize=(9, 7))
        gs = GridSpec(
            2, 2,
            figure=fig,
            width_ratios=[1.15, 1],
            height_ratios=[1, 1],
            wspace=0.35,
            hspace=0.35,
        )

        ax_profile = fig.add_subplot(gs[:, 0])  # spans both rows
        ax_eta = fig.add_subplot(gs[0, 1])
        ax_vdr = fig.add_subplot(gs[1, 1])
        # -----------------------------------------------------------------------------
        # Height-dependent VDR profile: left column, spanning both rows
        ax_profile.plot(vdr_inner, height, label="vdr_inner", c="k", zorder=3)
        ax_profile.plot(vdr_outer, height, c="grey", lw=1, zorder=-1)
        ax_profile.plot(vdr_outer_dash, height, c="grey", lw=1, zorder=-1)

        ax_profile.plot(
            gaussian_filter1d(vdr_outer, 5),
            height,
            label=f"vdr_outer HR {Hr_outer_init:.5g}",
        )

        ax_profile.plot(
            gaussian_filter1d(vdr_outer_dash, 5),
            height,
            label=f"vdr_outer HR {Hr:.5g}",
        )

        ax_profile.set_ylim(0, 12)
        ax_profile.set_xlim(0.003, 0.5)
        ax_profile.set_xscale("log")
        ax_profile.set_xlabel("voldepol")
        ax_profile.set_ylabel("Height [km]")
        ax_profile.legend(fontsize=8)
        ax_profile.set_title("VDR profiles", fontsize=10)

        # -----------------------------------------------------------------------------
        # Eta scatter: upper right
        ax_eta.plot(vdr_inner[height_mask], eta_outer[height_mask], ".", ms=3)
        ax_eta.axvline(vdr_inner_min, ls="--", c="dimgrey", lw=1)
        ax_eta.axhline(eta_outer_guess, lw=0.8, zorder=-1)

        ax_eta.set_xlim(0.0015, 0.8)
        ax_eta.set_xscale("log")
        ax_eta.set_xlabel("voldepol inner")
        ax_eta.set_ylabel("eta outer")
        ax_eta.set_title("Eta first guess", fontsize=10)

        # -----------------------------------------------------------------------------
        # Inner vs outer VDR scatter: lower right
        ax_vdr.plot([-3, 1], [-3, 1], color="k", ls="--", lw=1)

        ax_vdr.plot(
            gaussian_filter1d(vdr_inner[height_mask], 5),
            gaussian_filter1d(vdr_outer[height_mask], 5),
            ".", ms=3,
            label=(
                f"init: eta {eta_outer_guess:.2f} "
                f"HR {Hr_outer_init:.5g}"
            ),
        )

        ax_vdr.plot(
            gaussian_filter1d(vdr_inner[height_mask], 5),
            gaussian_filter1d(vdr_outer_dash[height_mask], 5),
            ".", ms=3,
            label=f"opt: eta {eta_outer_opt:.2f} HR {Hr:.5g}",
        )

        s = (
            f"Hr fixed {Hr_fixed}\n"
            f"rmse {rmse:.5f}\n"
            f"median ratio {median_ratio:.5f}\n"
        )
        ax_vdr.text(0.55, 0.02, s, transform=ax_vdr.transAxes, fontsize=8)

        ax_vdr.set_xlim(0.0015, 0.6)
        ax_vdr.set_ylim(0.0015, 0.6)
        ax_vdr.set_xscale("log")
        ax_vdr.set_yscale("log")
        ax_vdr.set_xlabel("voldepol inner")
        ax_vdr.set_ylabel("voldepol outer")
        ax_vdr.legend(fontsize=8)
        ax_vdr.set_title("Initial guess -> Optimized", fontsize=10)


        # Shared figure title and save
        fig.suptitle(s_title, fontsize=10, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        figname = f"{dt_grp:%Y%m%d_%H%M}_transfer_calibration.png"
        figs_out.append((fig, figname))

    return figs_out