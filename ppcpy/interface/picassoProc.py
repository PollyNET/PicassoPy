#from ..misc import *
import datetime
import re
import numpy as np
import logging
import ppcpy.misc.pollyChannelTags as pollyChannelTags
import ppcpy.preprocess.pollyPreprocess as pollyPreprocess
import ppcpy.qc.pollySaturationDetect as pollySaturationDetect
import ppcpy.qc.transCor as transCor
import ppcpy.qc.overlapEst as overlapEst
import ppcpy.qc.overlapCor as overlapCor


import ppcpy.calibration.polarization as polarization
import ppcpy.cloudmask.cloudscreen as cloudscreen
import ppcpy.cloudmask.profilesegment as profilesegment
import ppcpy.preprocess.profiles as preprocprofiles
import ppcpy.io.readMeteo as readMeteo
import ppcpy.misc.molecular as molecular
import ppcpy.calibration.rayleighfit as rayleighfit
import ppcpy.retrievals.klettfernald as klettfernald
import ppcpy.retrievals.raman as raman
import ppcpy.retrievals.depolarization as depolarization 
import ppcpy.retrievals.angstroem as angstroem 
import ppcpy.calibration.lidarconstant as lidarconstant

import ppcpy.retrievals.highres as highres
import ppcpy.retrievals.quasiV1 as quasiV1
import ppcpy.retrievals.quasiV2 as quasiV2
import ppcpy.retrievals.quasi as quasi

class PicassoProc:
    counter = 0

    def __init__(self, rawdata_dict, polly_config_dict, picasso_config_dict, polly_default_dict):
        type(self).counter += 1
        self.rawdata_dict = rawdata_dict
        self.polly_config_dict = polly_config_dict
        self.picasso_config_dict = picasso_config_dict
        self.polly_default_dict = polly_default_dict
        self.device = self.polly_config_dict['name']
        self.location = self.polly_config_dict['site']
        self.date = self.mdate_filename()
        self.num_of_channels = len(self.rawdata_dict['measurement_shots']['var_data'][0])
        self.num_of_profiles = self.rawdata_dict['raw_signal']['var_data'].shape[0]
        self.retrievals_highres = {}
        self.retrievals_profile = {}
        self.retrievals_profile['avail_optical_profiles'] = []

    def mdate_filename(self):
        filename = self.rawdata_dict['filename']
        mdate = re.split(r'_',filename)[0:3]
        YYYY = mdate[0]
        MM = mdate[1]
        DD = mdate[2]
        return f"{YYYY}{MM}{DD}"

    def gf(self, wavelength, meth, telescope):
        """get flag shorthand

        i.e., the following two calls are equivalent
        ```
        data_cube.flag_532_total_FR
        data_cube.gf(532, 'total', 'FR')
        ```        
        
        where the pattern `{wavelength}_{total|cross|parallel|rr}_{NR|FR|DFOV}` from
        https://github.com/PollyNET/Pollynet_Processing_Chain/issues/303 is obeyed

        Parameters
        ----------
        wavelength
            wavelength tag
        meth
            method
        telescope
            telescope

        Returns
        -------
        array
            with bool flag
        
        """
        return getattr(self, f'flag_{wavelength}_{meth}_{telescope}', False)

#    def msite(self):
#        #msite = f"measurement site: {self.rawdata_dict['global_attributes']['location']}"
#        msite = self.polly_config_dict['site']
#        logging.info(f'measurement site: {msite}')
#        return msite
#
#    def device(self):
#        device = self.polly_config_dict['name']
#        logging.info(f'measurement device: {device}')
#        return device
#
#    def mdate(self):
#        mdate = self.mdate_filename()
#        logging.info(f'measuremnt date: {mdate}')
#        return mdate

    def mdate_infile(self):
        mdate_infilename = self.rawdata_dict['measurement_time']['var_data'][0][0]
        return f"{mdate_infilename}"

    def check_for_correct_mshots(self):
        laser_rep_rate = self.rawdata_dict['laser_rep_rate']['var_data']
        mShotsPerPrf = laser_rep_rate * self.polly_config_dict['deltaT']
        mShots = self.rawdata_dict['measurement_shots']['var_data']

        # Check for values > 1.1*mShotsPerPrf or <= 0
        condition_check_matrix = (mShots > mShotsPerPrf*1.1) | (mShots <= 0)
        
        return condition_check_matrix

    def filter_or_correct_false_mshots(self):
        logging.info(f"flagFilterFalseMShots: {self.polly_config_dict['flagFilterFalseMShots']}")
        logging.info(f"flagCorrectFalseMShots: {self.polly_config_dict['flagCorrectFalseMShots']}")

        if self.polly_config_dict['flagFilterFalseMShots']:
            logging.info('filtering false mshots')
        elif self.polly_config_dict['flagCorrectFalseMShots']:
            logging.info('correcting false mshots')

        condition_check_matrix = self.check_for_correct_mshots()
        ##TODO
        return self

    def mdate_consistency(self) -> bool:
        if self.mdate_filename() == self.mdate_infile():
            logging.info('... date in nc-file equals date of filename')
            return True
        else:
            logging.warning('... date in nc-file differs from date of filename')
            return False

    def reset_date_infile(self):
        logging.info('date consistency-check... ')
        if self.mdate_consistency() == False:
            logging.info('date in nc-file will be replaced with date of filename.')
            np_array = np.array(self.rawdata_dict['measurement_time']['var_data']) ## converting to numpy-array for easier numpy-operations
            mdate = self.mdate_filename()
            np_array[:,0] = mdate ## assign new date value to the whole first column of the 2d-numpy-array
            self.rawdata_dict['measurement_time']['var_data'] = np_array
            return self
        else:
            pass

    def setChannelTags(self):
        ChannelTags = pollyChannelTags.pollyChannelTags(self.polly_config_dict['channelTag'],flagFarRangeChannel=self.polly_config_dict['isFR'], ##TODO key: channelTags vs channelTag???
                                                                               flagNearRangeChannel=self.polly_config_dict['isNR'],
                                                                               flagRotRamanChannel=self.polly_config_dict['isRR'],
                                                                               flagTotalChannel=self.polly_config_dict['isTot'],
                                                                               flagCrossChannel=self.polly_config_dict['isCross'],
                                                                               flagParallelChannel=self.polly_config_dict['isParallel'],
                                                                               flag355nmChannel=self.polly_config_dict['is355nm'],
                                                                               flag387nmChannel=self.polly_config_dict['is387nm'],
                                                                               flag407nmChannel=self.polly_config_dict['is407nm'],
                                                                               flag532nmChannel=self.polly_config_dict['is532nm'],
                                                                               flag607nmChannel=self.polly_config_dict['is607nm'],
                                                                               flag1064nmChannel=self.polly_config_dict['is1064nm']
                                                                               )

        ChannelTags, self.polly_config_dict = pollyChannelTags.polly_config_channel_corrections(chTagsOut_ls=ChannelTags,polly_config_dict=self.polly_config_dict)

        self.retrievals_highres['channel'] = ChannelTags
        self.polly_config_dict['channelTags'] = ChannelTags
        self.channel_dict = {i: item for i, item in enumerate(ChannelTags)}

        ChannelFlags =  pollyChannelTags.pollyChannelflags(channel_dict_length=len(self.channel_dict),flagFarRangeChannel=self.polly_config_dict['isFR'],
                                                                               flagNearRangeChannel=self.polly_config_dict['isNR'],
                                                                               flagRotRamanChannel=self.polly_config_dict['isRR'],
                                                                               flagTotalChannel=self.polly_config_dict['isTot'],
                                                                               flagCrossChannel=self.polly_config_dict['isCross'],
                                                                               flagParallelChannel=self.polly_config_dict['isParallel'],
                                                                               flag355nmChannel=self.polly_config_dict['is355nm'],
                                                                               flag387nmChannel=self.polly_config_dict['is387nm'],
                                                                               flag407nmChannel=self.polly_config_dict['is407nm'],
                                                                               flag532nmChannel=self.polly_config_dict['is532nm'],
                                                                               flag607nmChannel=self.polly_config_dict['is607nm'],
                                                                               flag1064nmChannel=self.polly_config_dict['is1064nm']
                                                                               )
        
        self.flags = ChannelFlags
        self.flag_355_total_FR = ChannelFlags[0]
        self.flag_355_cross_FR = ChannelFlags[1]
        self.flag_355_parallel_FR = ChannelFlags[2]
        self.flag_355_total_NR = ChannelFlags[3]
        self.flag_387_total_FR = ChannelFlags[4]
        self.flag_387_total_NR = ChannelFlags[5]
        self.flag_407_total_FR = ChannelFlags[6]
        self.flag_407_total_NR = ChannelFlags[7]
        self.flag_532_total_FR = ChannelFlags[8]
        self.flag_532_cross_FR = ChannelFlags[9]
        self.flag_532_parallel_FR = ChannelFlags[10]
        self.flag_532_total_NR = ChannelFlags[11]
        self.flag_532_cross_DFOV = ChannelFlags[12]
        self.flag_532_rr_FR = ChannelFlags[13]
        self.flag_607_total_FR = ChannelFlags[14]
        self.flag_607_total_NR = ChannelFlags[15]
        self.flag_1058_total_FR = ChannelFlags[16]
        self.flag_1064_total_FR = ChannelFlags[17]
        self.flag_1064_cross_FR = ChannelFlags[18]
        self.flag_1064_total_NR = ChannelFlags[19]


        return self

    def preprocessing(self, collect_debug=False):
        preproc_dict = pollyPreprocess.pollyPreprocess(self.rawdata_dict,
                deltaT=self.polly_config_dict['deltaT'],
                flagForceMeasTime = self.polly_config_dict['flagForceMeasTime'],
                maxHeightBin = self.polly_config_dict['max_height_bin'],
                firstBinIndex = self.polly_config_dict['first_range_gate_indx'],
                firstBinHeight = self.polly_config_dict['first_range_gate_height'],
                pollyType = self.polly_config_dict['name'],
                flagDeadTimeCorrection = self.polly_config_dict['flagDTCor'],
                deadtimeCorrectionMode = self.polly_config_dict['dtCorMode'],
                deadtimeParams = self.polly_config_dict['dt'],
                flagSigTempCor = self.polly_config_dict['flagSigTempCor'],
                tempCorFunc = self.polly_config_dict['tempCorFunc'],
                meteorDataSource = self.polly_config_dict['meteorDataSource'],
                gdas1Site = self.polly_config_dict['gdas1Site'],
                gdas1_folder = self.picasso_config_dict['gdas1_folder'],
                radiosondeSitenum = self.polly_config_dict['radiosondeSitenum'],
                radiosondeFolder = self.polly_config_dict['radiosondeFolder'],
                radiosondeType = self.polly_config_dict['radiosondeType'],
                bgCorrectionIndex = self.polly_config_dict['bgCorRangeIndx'],
                asl = self.polly_config_dict['asl'],
                initialPolAngle = self.polly_config_dict['init_depAng'],
                maskPolCalAngle = self.polly_config_dict['maskDepCalAng'],
                minSNRThresh = self.polly_config_dict['mask_SNRmin'],
                minPC_fog = self.polly_config_dict['minPC_fog'],
                flagFarRangeChannel = self.polly_config_dict['isFR'],
                flag532nmChannel = self.polly_config_dict['is532nm'],
                flagTotalChannel = self.polly_config_dict['isTot'],
                flag355nmChannel = self.polly_config_dict['is355nm'],
                flag607nmChannel = self.polly_config_dict['is607nm'],
                flag387nmChannel = self.polly_config_dict['is387nm'],
                flag407nmChannel = self.polly_config_dict['is407nm'],
                flag355nmRotRaman = np.bitwise_and(np.array(self.polly_config_dict['is355nm']), np.array(self.polly_config_dict['isRR'])).tolist(),
                flag532nmRotRaman = np.bitwise_and(np.array(self.polly_config_dict['is532nm']), np.array(self.polly_config_dict['isRR'])).tolist(),
                flag1064nmRotRaman = np.bitwise_and(np.array(self.polly_config_dict['is1064nm']), np.array(self.polly_config_dict['isRR'])).tolist(),
                isUseLatestGDAS = self.polly_config_dict['flagUseLatestGDAS'],
                collect_debug=collect_debug,
                )
        self.retrievals_highres.update(preproc_dict)

        return self

    def SaturationDetect(self):

        self.flagSaturation = pollySaturationDetect.pollySaturationDetect(
            data_cube = self,
            sigSaturateThresh = self.polly_config_dict['saturate_thresh'])

        return self


    def polarizationCaliD90(self):
        """
        
        The stuff that starts here in the matlab version
        https://github.com/PollyNET/Pollynet_Processing_Chain/blob/5efd7d35596c67ef8672f5948e47d1f9d46ab867/lib/interface/picassoProcV3.m#L442
        """

        polarization.loadGHK(self)
        self.pol_cali = polarization.calibrateGHK(self)


    def cloudScreen(self):
        """https://github.com/PollyNET/Pollynet_Processing_Chain/blob/b3b8ec7726b75d9db6287dcba29459587ca34491/lib/interface/picassoProcV3.m#L663"""
        self.flagCloudFree = cloudscreen.cloudscreen(self)


    def cloudFreeSeg(self):
        """https://github.com/PollyNET/Pollynet_Processing_Chain/blob/b3b8ec7726b75d9db6287dcba29459587ca34491/lib/interface/picassoProcV3.m#L707
        
        .. code-block:: python
        
            data_cube.clFreGrps = [
                [35,300],
                [2500,2800]
            ]
        
        """
        self.clFreeGrps = profilesegment.segment(self)

    def aggregate_profiles(self, var=None):
        """
        
        """

        if var == None:
            self.retrievals_profile['RCS'] = \
                preprocprofiles.aggregate_clFreeGrps(self, 'RCS', func=np.nanmean)
            self.retrievals_profile['sigBGCor'] = \
                preprocprofiles.aggregate_clFreeGrps(self, 'sigBGCor')
            self.retrievals_profile['BG'] = \
                preprocprofiles.aggregate_clFreeGrps(self, 'BG')

        else:
            self.retrievals_profile[var] = \
                preprocprofiles.aggregate_clFreeGrps(self, var)


    def loadMeteo(self):

        self.met = readMeteo.Meteo(
            self.polly_config_dict['meteorDataSource'], 
            self.polly_config_dict['meteo_folder'],
            self.polly_config_dict['meteo_file'])
        self.met.load(
            datetime.datetime.timestamp(datetime.datetime.strptime(self.date, '%Y%m%d')),
            self.retrievals_highres['height'])


    def loadAOD(self):
        """"""
        pass


    def calcMolecular(self):
        """calculate the molecular scattering for the cloud free periods
        
        with the strategy of first averaging the met data and then calculating the rayleigh scattering
        
        """

        time_slices = [self.retrievals_highres['time64'][grp] for grp in self.clFreeGrps]
        print('time slices of cloud free ', time_slices)
        mean_profiles = self.met.get_mean_profiles(time_slices) 
        self.mol_profiles = molecular.calc_profiles(mean_profiles)
    

    def rayleighFit(self):
        """do the rayleigh fit
        
        direct translation from the matlab code. There might be noticeable numerical discrepancies (especially in the residual)
        seemed to work ok for 532, 1064, but with issues for 355
        """

        print('Start Rayleigh Fit')
        logging.warning(f'Potential for differences to matlab code du to numerical issues (subtraction of two small values)')

        self.refH =  rayleighfit.rayleighfit(self)
        return self.refH


    def polarizationCaliMol(self):
        """
        
        """

        logging.warning(f'not checked against the matlab code')
        if self.polly_config_dict['flagMolDepolCali']:
            self.pol_cali_mol = polarization.calibrateMol(self)


    def transCor(self):
        """
        
        """

        if self.polly_config_dict['flagTransCor']:
            logging.warning('transmission correction')
            self.retrievals_highres['sigTCor'], self.retrievals_highres['BGTCor'] = \
                  transCor.transCorGHK_cube(self)
        else:
            logging.warning('NO transmission correction')


    def retrievalKlett(self, oc=False, nr=False):
        """
        """

        retrievalname = 'klett'
        kwargs = {}
        if oc:
            retrievalname +='_OC'
            kwargs['signal'] = 'OLCor'
        if nr:
            kwargs['nr'] = True

        print('retrievalname', retrievalname)
        self.retrievals_profile[retrievalname] = \
            klettfernald.run_cldFreeGrps(self, **kwargs)
        if retrievalname not in self.retrievals_profile['avail_optical_profiles']:
            self.retrievals_profile['avail_optical_profiles'].append(retrievalname)


    def retrievalRaman(self, oc=False, nr=False):
        """
        """

        retrievalname = 'raman'
        kwargs = {}
        if oc:
            retrievalname +='_OC'
            kwargs['signal'] = 'OLCor'

            # get the full overlap height for the overlap corrected variant
            # group by the cloud free groups 
            kwargs['heightFullOverlap'] = \
                [np.mean(self.retrievals_highres['heightFullOverCor'][slice(*cF)], axis=0) for 
                 cF in self.clFreeGrps]
        if nr:
            kwargs['nr'] = True

        self.retrievals_profile[retrievalname] = \
            raman.run_cldFreeGrps(self, **kwargs)
        if retrievalname not in self.retrievals_profile['avail_optical_profiles']:
            self.retrievals_profile['avail_optical_profiles'].append(retrievalname)


    def overlapCalc(self):
        """estimate the overlap function

        different to the matlab version, where an average over all cloud
        free periods is taken, it is done here per cloud free segment
            
        """

        self.retrievals_profile['overlap'] = {}
        self.retrievals_profile['overlap']['frnr'] = overlapEst.run_frnr_cldFreeGrps(self)
        self.retrievals_profile['overlap']['raman'] = overlapEst.run_raman_cldFreeGrps(self)
    
    def overlapFixLowestBins(self):
        """the lowest bins are affected by stange near range effects"""

        height = self.retrievals_highres['range']
        for k in self.retrievals_profile['overlap']:
            return overlapCor.fixLowest(
                self.retrievals_profile['overlap'][k], np.where(height > 800)[0][0])


    def overlapCor(self):
        """

        the overlap correction is implemented differently to the matlab version
        first a 2d (time, height) correction array is constructed then it is applied.
        In future this will allow for time variing overlap functions
        
        """

        if self.polly_config_dict['overlapCorMode'] == 0:
            logging.info('no overlap Correction')
            return self
        logging.info('overlap Correction')
        if self.polly_config_dict['overlapCorMode'] == 1:
            logging.info('overlapCorMode 1 -> need file for overlapfunction')
            self.retrievals_profile['overlap']['file'] = overlapEst.load(self)
        self.retrievals_highres['overlap2d'] = overlapCor.spread(self)
        ret = overlapCor.apply_cube(self)
        self.retrievals_highres['sigOLCor'] = ret[0]
        self.retrievals_highres['BGOLCor'] = ret[1]
        self.retrievals_highres['heightFullOverCor'] = ret[2]


    def calcDepol(self):
        """
        """
        
        for ret_prof_name in self.retrievals_profile['avail_optical_profiles']:
            print(ret_prof_name)
        
            self.retrievals_profile[ret_prof_name] = depolarization.voldepol_cldFreeGrps(
                self, ret_prof_name) 
            self.retrievals_profile[ret_prof_name] = depolarization.pardepol_cldFreeGrps(
                self, ret_prof_name) 


    def Angstroem(self):
        """
        """
        for ret_prof_name in self.retrievals_profile['avail_optical_profiles']:
            print(ret_prof_name)
        
            self.retrievals_profile[ret_prof_name] = angstroem.ae_cldFreeGrps(
                self, ret_prof_name) 

    def LidarCalibration(self):
        """
        """
        self.LC = {}
        self.LC['klett'] = lidarconstant.lc_for_cldFreeGrps(
            self, 'klett')
        self.LC['raman'] = lidarconstant.lc_for_cldFreeGrps(
            self, 'raman')
        
        logging.warning('reading calibration constant from database not working yet')
        self.LCused = lidarconstant.get_best_LC(self.LC['raman'])


    def attBsc_volDepol(self):
        """highres attBsc and voldepol in 2d
        """

        # for now try with mutable state in data_cube
        logging.info('attBsc 2d retrieval')
        highres.attbsc_2d(self)

        logging.info('voldepol 2d retrieval')
        highres.voldepol_2d(self)


    def molecularHighres(self):
        """
        """

        self.mol_2d = molecular.calc_2d(
            self.met.ds)


    def quasiV1(self):
        """
        """

        quasiV1.quasi_bsc(self)
        quasi.quasi_pdr(self, version='V1')
        quasi.quasi_angstrom(self, version='V1')
        quasi.target_cat(self, version='V1')

    def quasiV2(self):
        """
        """

        quasiV2.quasi_bsc(self)
        quasi.quasi_pdr(self, version='V2')
        quasi.quasi_angstrom(self, version='V2')
        quasi.target_cat(self, version='V2')


#    def __str__(self):
#        return f"{self.rawdata_dict}"

    def __del__(self):
        type(self).counter -= 1
        

