


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_optical_analysis(profiles, height, times, plot_settings={}, smooth=None, refH=None):
    """quick first version of the optical profiles plotting function
     
    """
    fig, ax = plt.subplots(1, 5, figsize=[12, 6.5], sharey=True)

    # backscatter
    height = height.copy()/1000.
    h_top = 20
    ihtop = np.where(height > h_top)[0][0]
    ax[0].axvline(0, lw=0.8, c='k')
    if '355_total_NR' in profiles and 'aerBsc' in profiles['355_total_NR']:
        ax[0].plot(profiles['355_total_NR']['aerBsc'][:ihtop]*1e6, height[:ihtop], 
                   color='skyblue', label='355NR')
    else:
        print('near range bsc issue 355')
    if '532_total_NR' in profiles and 'aerBsc' in profiles['532_total_NR']:
        ax[0].plot(profiles['532_total_NR']['aerBsc'][:ihtop]*1e6, height[:ihtop], 
                   color='lawngreen', label='532NR')
    else:
        print('near range bsc issue 532')

    ax[0].plot(profiles['355_total_FR']['aerBsc']*1e6, height, 
               color='blue', label='355')
    d_err = 75
#     ax[0].errorbar(data[::d_err,1], data[::d_err,0], 
#                    xerr=data[::d_err,2],
#                    fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#                    color='blue')
    ax[0].plot(profiles['532_total_FR']['aerBsc']*1e6, height, 
               color='green', label='532')
#     ax[0].errorbar(data[15::d_err,3], data[15::d_err,0], 
#                xerr=data[15::d_err,4],
#                fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#                color='green')
    ax[0].plot(profiles['1064_total_FR']['aerBsc']*1e6, height, 
               color='red', label='1604')
#     ax[0].errorbar(data[30::d_err,5], data[30::d_err,0], 
#                xerr=data[30::d_err,6],
#                fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#                color='red')

    # extinction
    ax[1].axvline(0, lw=0.8, c='k')
    if 'ext_nf_top' in plot_settings and plot_settings['ext_nf_top'] != -1: 
        h_top = plot_settings['ext_nf_top']
        i_nr_top = np.where(height > h_top)[0][0]
    else:
        i_nr_top = -1
    if 'ext_nf_base' in plot_settings and plot_settings['ext_nf_base'] != 0: 
        h_base = plot_settings['ext_nf_base']
        i_nr_base = np.where(height > h_base)[0][0]
    else:
        i_nr_base = 0
    if i_nr_top != i_nr_base and '355_total_NR' in profiles and 'aerExt' in profiles['355_total_NR']:
        ax[1].plot(profiles['355_total_NR']['aerExt'][i_nr_base:i_nr_top]*1e6, height[i_nr_base:i_nr_top], 
                   color='skyblue', label='355NR')
    if i_nr_top != i_nr_base and '532_total_NR' in profiles and 'aerExt' in profiles['532_total_NR']:
        ax[1].plot(profiles['532_total_NR']['aerExt'][i_nr_base:i_nr_top]*1e6, height[i_nr_base:i_nr_top],
                   color='lawngreen', label='532NR')

    if 'ext_ff_top' in plot_settings and plot_settings['ext_ff_top'] != -1: 
        h_top = plot_settings['ext_ff_top']
        i_fr_top = np.where(height > h_top)[0][0]
    else:
        i_fr_top = -1
    if 'ext_ff_base' in plot_settings and plot_settings['ext_ff_base'] != 0: 
        h_base = plot_settings['ext_ff_base']
        i_fr_base = np.where(height > h_base)[0][0]
    else:
        i_fr_base = 0   
    print(i_fr_base, i_fr_top)
    ax[1].plot(profiles['355_total_FR']['aerExt'][i_fr_base:i_fr_top]*1e6, height[i_fr_base:i_fr_top],
               color='blue', label='355', lw=1.2)
#     ax[1].errorbar(data[i_fr_base:i_fr_top:d_err,7], data[i_fr_base:i_fr_top:d_err,0], 
#                xerr=data[i_fr_base:i_fr_top:d_err,8],
#                fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#                color='blue')
    ax[1].plot(profiles['532_total_FR']['aerExt'][i_fr_base:i_fr_top]*1e6, height[i_fr_base:i_fr_top],
               color='green', label='532', lw=1.2)
#     ax[1].errorbar(data[i_fr_base:i_fr_top,9][20::d_err], data[i_fr_base:i_fr_top,0][20::d_err], 
#                xerr=data[i_fr_base:i_fr_top,10][20::d_err],
#                fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#                color='green')
    ax[1].plot(profiles['1064_total_FR']['aerExt'][i_fr_base:i_fr_top]*1e6, height[i_fr_base:i_fr_top],
               color='red', label='1064', lw=1.2)

    if profiles['532_total_FR']['retrieval'] == 'raman':
        ax[2].axvline(0, lw=0.8, c='k')
        # lidar ratio
        if 'lr_nf_top' in plot_settings and plot_settings['lr_nf_top'] != -1: 
            h_top = plot_settings['lr_nf_top']
            i_nr_top = np.where(height > h_top)[0][0]
        else:
            i_nr_top_355 = -1
            i_nr_top_532 = -1
        if 'lr_nf_base' in plot_settings and plot_settings['lr_nf_base'] != 0: 
            h_base = plot_settings['lr_nf_base']
            i_nf_base = np.where(height > h_base)[0][0]
        else:
            i_nf_base = 0 
        #print('ind', i_nr_top_355, i_nr_top_532, i_nf_base)

        if i_nr_top != i_nf_base:
            if '355_total_NR' in profiles and 'aerBsc' in profiles['355_total_NR']:
                ax[2].plot(profiles['355_total_NR']['LR'][i_nr_base:i_nr_top], height[i_nr_base:i_nr_top], 
                           color='skyblue', label='355NR')
            else:
                print('near range LR 355 issue')                
            if '532_total_NR' in profiles and 'aerBsc' in profiles['532_total_NR']:
                ax[2].plot(profiles['532_total_NR']['LR'][i_nr_base:i_nr_top], height[i_nr_base:i_nr_top], 
                           color='lawngreen', label='532NR')
            else:
                print('near range LR 532 issue')

        if 'lr_ff_top' in plot_settings and plot_settings['lr_ff_top'] != -1: 
            h_top = plot_settings['ext_ff_top']
            i_fr_top = np.where(height > h_top)[0][0]
        else:
            i_fr_top = -1
        if 'lr_ff_base' in plot_settings and plot_settings['lr_ff_base'] != 0: 
            h_base = plot_settings['lr_ff_base']
            i_fr_base = np.where(height > h_base)[0][0]
        else:
            i_fr_base = 0   
        ax[2].plot(profiles['355_total_FR']['LR'][i_fr_base:i_fr_top], height[i_fr_base:i_fr_top],
                   color='blue', label='355', lw=1.2)
        d_err = 75
    #     ax[2].errorbar(data[i_fr_base:i_fr_top:d_err,11], data[i_fr_base:i_fr_top:d_err,0], 
    #                    xerr=data[i_fr_base:i_fr_top:d_err,12],
    #                    fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
    #                    color='blue')
        ax[2].plot(profiles['532_total_FR']['LR'][i_fr_base:i_fr_top], height[i_fr_base:i_fr_top],
                   color='green', label='532', lw=1.2)
    #     ax[2].errorbar(data[i_fr_base:i_fr_top,13][20::d_err], data[i_fr_base:i_fr_top,0][20::d_err], 
    #                    xerr=data[i_fr_base:i_fr_top,14][20::d_err],
    #                    fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
    #                    color='green')


    # angstrom
    h_top = 6
    ihtop = np.where(height > h_top)[0][0]
    if '355_total_NR' in profiles and 'AE_Bsc_355_532' in profiles['355_total_NR']:
        ax[3].plot(profiles['355_total_NR']['AE_Bsc_355_532'][:ihtop], height[:ihtop],
                   color='skyblue', label='355/532NR')

    ax[3].plot(profiles['355_total_FR']['AE_Bsc_355_532'], height, 
               color='blue', label='355/532')
#     ax[3].errorbar(data[::d_err,17], data[::d_err,0], 
#                xerr=data[::d_err,18],
#                fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#                color='blue')
    ax[3].plot(profiles['532_total_FR']['AE_Bsc_532_1064'], height, 
               color='green', label='532/1064')
#     ax[3].errorbar(data[20::d_err,19], data[20::d_err,0], 
#                xerr=data[20::d_err,20],
#                fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#                color='green')
    if profiles['355_total_FR']['retrieval'] == 'raman' and not np.all(np.isnan(profiles['355_total_FR']['AE_Ext_355_532'])):
        ax[3].plot(profiles['355_total_FR']['AE_Ext_355_532'], height, 
           color='steelblue', label='Ext355/532')
    if '355_total_NR' in profiles and profiles['355_total_NR']['retrieval'] == 'raman' and not np.all(np.isnan(profiles['355_total_NR']['AE_Ext_355_532'])):
        ax[3].plot(profiles['355_total_NR']['AE_Ext_355_532'], height, 
           color='darkblue', label='Ext355/532 NR')

    # depol
    ax[4].axvline(0, lw=0.8, c='k')
    ax[4].plot(profiles['355_total_FR']['vdr'], height, 
               color='lightblue', label='δvol 355')
    ax[4].plot(profiles['532_total_FR']['vdr'], height, 
               color='lightgreen', label='δvol 532')
    ax[4].plot(profiles['355_total_FR']['pdr'], height, 
               color='blue', label='δpar 355')
#     ax[4].errorbar(data[::d_err,34], data[::d_err,33], 
#            xerr=data[::d_err,35],
#            fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#            color='blue')
    ax[4].plot(profiles['532_total_FR']['pdr'], height, 
               color='green', label='δpar 532')
#     ax[4].errorbar(data[20::d_err,25], data[20::d_err,24], 
#            xerr=data[20::d_err,26],
#            fmt='.', capsize=5, linewidth=1.5, elinewidth=1.5,
#            color='green')
    
    # axis labels and stuff
    if 'bsc_lims' in plot_settings:
        ax[0].set_xlim(plot_settings['bsc_lims'])
    else:
        ax[0].set_xlim([-0.1, 0.5])
    ax[0].set_ylabel("Height [km]", fontsize=13.5)
    ax[0].set_xlabel("Bsc. coeff.\n[Mm$^{-1}$ sr$^{-1}$]", fontsize=13.5)

    if 'ext_lims' in plot_settings:
        ax[1].set_xlim(plot_settings['ext_lims'])
    else:
        ax[1].set_xlim([-5, 15])
    ax[1].set_xlabel("Ext. coeff.\n[Mm$^{-1}$]", fontsize=13.5)

#     ax[2].set_xlim(0, 100)
    ax[2].set_xlim(-100, 150)
    ax[2].set_xlabel("Lidar ratio [sr]", fontsize=13.5)

    ax[3].set_xlim(-0.5, 5)
    ax[3].set_xlabel("Ångström exp.", fontsize=13.5)

    if 'dep_lims' in plot_settings:
        ax[4].set_xlim(plot_settings['dep_lims'])
    else:
        ax[4].set_xlim(-0.05, 0.35)
    ax[4].set_xlabel("Depol. ratio", fontsize=13.5)
    
    if 'plottop' in plot_settings:
        ax[0].set_ylim([0, plot_settings['plottop']])
    else:
        ax[0].set_ylim([0,22])


    string = "{} - {} {}".format(
        times[0].strftime("%Y%m%d %H:%M"), 
        times[1].strftime("%H:%M"),
        profiles['355_total_FR']['retrieval'])
    fig.suptitle(string, fontsize=14)

    #if 'smooth532' in data.keys():
    #    string = f"smooth 532: {data['smooth532']}, 532NR: {data['smooth532NR']} bins"
    #    fig.text(0.65, 0.95, string,
    #            transform=fig.transFigure)

    for i in range(5):
        ax[i].legend(loc='upper right', fontsize=10)
        ax[i].tick_params(axis='both', which='both', right=True, top=True)
        ax[i].tick_params(axis='both', which='major', labelsize=13,
                       width=2, length=5.5)
        ax[i].tick_params(axis='both', which='minor', width=1.3, length=3)
        ax[i].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    fig.tight_layout()
    fig.subplots_adjust(top=0.93, wspace = 0.2)
    
    return fig, ax