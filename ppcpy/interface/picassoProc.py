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
import ppcpy.qc.qualityMask as qualityMask

import ppcpy.calibration.select as select
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

import ppcpy.io.sql_interaction as sql_db


class PicassoProc:
    """Picasso Processor.

    This class is responsible for perfoming the processing of PollyXT data.
    """
    counter = 0

    def __init__(self, rawdata_dict:dict, polly_config_dict:dict, picasso_config_dict:dict):
        """Initialize the PicassoProc object.

        Parameters
        ----------
        rawdata_dict : dict
            The dict returned by readPollyRawData.readPollyRawData(filename=rawfile).
        polly_config_dict : dict
            The configuration specific to the specific polly loadConfigs.loadPollyConfig(polly_config_file_fullname, polly_default_config_file).
        picasso_config_dict : dict
            The general picasso config loadConfigs.loadPicassoConfig(args.picasso_config_file,picasso_default_config_file).

        Yields
        ------
        self.rawfile : str
            Path to level0-file.
        self.rawdata_dict : dict
            The input parameter `rawdata_dict`.
        self.polly_config_dict : dict
            The input parameter `polly_config_dict`.
        self.picasso_config_dict : dict
            The input parameter `picasso_config_dict`.
        self.device : str
            Name of pollyXT device.
        self.location : ...
            Location of PollyXT device during the measurement.
        self.date : str
            Measurement date.
        self.num_of_channels : int
            Number of measurment channels.
        self.num_of_profiles : int
            Number of profiles in the measurment (time-dimension).
        self.retrievals_highres : dict
            Dictionary to store high resolution time, height data.
        self.retrievals_profile : dict
            Dictionary to store profile date.
        self.retrievals_profile['avail_optical_profiles'] : list
            Availabele optical profiles.
        self.pol_cali : dict
            Dictionary to store polarization calibration constants.
        self.LC : dict
            Dictionary to store lidar calibration constants.

        Notes
        -----
        - The `polly_default_dict` is not longer available as a separate variable, but is included into the `polly_config_dict`.
        """

        type(self).counter += 1
        self.rawfile = rawdata_dict['filename_path']
        self.rawdata_dict = rawdata_dict
        self.polly_config_dict = polly_config_dict
        self.picasso_config_dict = picasso_config_dict
        self.device = self.polly_config_dict['name']
        self.location = self.polly_config_dict['site']
        self.date = self.mdate_filename()
        self.num_of_channels = len(self.rawdata_dict['measurement_shots']['var_data'][0])
        self.num_of_profiles = self.rawdata_dict['raw_signal']['var_data'].shape[0]
        self.retrievals_highres = {}
        self.retrievals_profile = {}
        self.retrievals_profile['avail_optical_profiles'] = []
        self.pol_cali = {}
        self.LC = {}


    def mdate_filename(self) -> str:
        """Get the date from filename in YYYYMMDD.
        
        Returns
        -------
        str
            Measurement date.
        """

        filename = self.rawdata_dict['filename']
        mdate = re.split(r'_', filename)[0:3]
        YYYY = mdate[0]
        MM = mdate[1]
        DD = mdate[2]
        return f"{YYYY}{MM}{DD}"


    def gf(self, wavelength:float|int|str, meth:str, telescope:str) -> np.ndarray:
        """Get flag shorthand.

        i.e., the following two calls are equivalent
        ```
        data_cube.flag_532_total_FR
        data_cube.gf(532, 'total', 'FR')
        ```        
        
        where the pattern `{wavelength}_{total|cross|parallel|rr}_{NR|FR|DFOV}` from
        https://github.com/PollyNET/Pollynet_Processing_Chain/issues/303 is obeyed

        Parameters
        ----------
        wavelength : float or int or str
            Wavelength tag.
        meth : str
            Method type. 
        telescope : str
            Telescope type.

        Returns
        -------
        ndarray
            With bool flag
        """

        return getattr(self, f'flag_{wavelength}_{meth}_{telescope}', False)


#    def msite(self):
#        #msite = f"measurement site: {self.rawdata_dict['global_attributes']['location']}"
#        msite = self.polly_config_dict['site']
#        logging.info(f'measurement site: {msite}')
#        return msite
#
#
#    def device(self):
#        device = self.polly_config_dict['name']
#        logging.info(f'measurement device: {device}')
#        return device
#
#
#    def mdate(self):
#        mdate = self.mdate_filename()
#        logging.info(f'measuremnt date: {mdate}')
#        return mdate


    def mdate_infile(self) -> str:
        """First date in file as string.
        
        Returns
        -------
        str
            Measurement date.
        """

        mdate_infilename = self.rawdata_dict['measurement_time']['var_data'][0][0]
        return f"{mdate_infilename}"


    def check_for_correct_mshots(self) -> np.ndarray:
        """Check if mshots are more than 1.1 * laser_prf * deltatime or smaller 0.
        
        Returns
        -------
        condition_check_matrix : ndarray
            Boolean array. True for elements where the condition above holds, otherwise False.
        """

        laser_rep_rate = self.rawdata_dict['laser_rep_rate']['var_data']
        mShotsPerPrf = laser_rep_rate * self.polly_config_dict['deltaT']
        mShots = self.rawdata_dict['measurement_shots']['var_data']

        # Check for values > 1.1*mShotsPerPrf or <= 0
        condition_check_matrix = (mShots > mShotsPerPrf*1.1) | (mShots <= 0)
        
        return condition_check_matrix


    def filter_or_correct_false_mshots(self):
        """Filter or correct the mshots (currently only logging).
        
        .. TODO::
            that might be covered via the mcps conversion
            --> How exactly???
        
        Yields
        ------
        self.rawdata_dict : dict
            With filtered or corrected enteries for timestamps where `data_cube.check_for_correct_mshots()=True`.
        
        Notes
        -----
        - Now includes a first try for the function. I Do not know if it actually is needed yet.

        .. TODO:: Not yet crosschecked with the matlab version.

        ** History **
        
        - 2026-06-24: Translated to python
        
        """
        
        ## Extract the necesary information
        laser_rep_rate = self.rawdata_dict['laser_rep_rate']['var_data']
        mShotsPerPrf = laser_rep_rate * self.polly_config_dict['deltaT']
        mShots = self.rawdata_dict['measurement_shots']['var_data']
        mTime = self.rawdata_dict['measurement_time']['var_data']
        rawSig = self.rawdata_dict['raw_signal']['var_data']

        condition_check_matrix = self.check_for_correct_mshots()
        condition_check_matrix = np.any(condition_check_matrix, axis=0)

        ## Filter out timesteps with faulty measurement shots
        if self.polly_config_dict['flagFilterFalseMShots']:
            logging.info('filtering false mshots')

            if np.sum(~condition_check_matrix) == 0:
                logging.critical('No profile with mshots < 1e6 and mshots > 0 was found Please take a look inside level0 file.')
                # raise ValueError('No profile with mshots < 1e6 and mshots > 0 was found Please take a look inside level0 file.')
                return
            
            rawSig = rawSig[~condition_check_matrix]
            mShots = mShots[~condition_check_matrix]
            mTime = mTime[~condition_check_matrix]
            if 'depol_cal_angle' in self.rawdata_dict:
                self.rawdata_dict['depol_cal_angle']['var_data'] \
                    = self.rawdata_dict['depol_cal_angle']['var_data'][~condition_check_matrix]

        ## Correct timesteps with faulty measurement shots
        elif self.polly_config_dict['flagCorrectFalseMShots']:
            logging.info('correcting false mshots')

            mShots[condition_check_matrix] = mShotsPerPrf
            # .. TODO:: In the matlab vesion this also does a fix to mTime. I do not know if this is necesary yet. 
        
        ## Overwrite the filtered / corrected information
        self.rawdata_dict['measurement_shots']['var_data'] = mShots
        self.rawdata_dict['measurement_time']['var_data'] = mTime
        self.rawdata_dict['raw_signal']['var_data'] = rawSig


    def mdate_consistency(self) -> bool:
        """Check mdate consistency.
        
        Returns
        -------
        bool
            True, if measurement date of file name equals
            that of in the file, otherwise False.
        """

        if self.mdate_filename() == self.mdate_infile():
            logging.info('... date in nc-file equals date of filename')
            return True
        else:
            logging.warning('... date in nc-file differs from date of filename')
            return False


    def reset_date_infile(self):
        """Correct the date in the file.

        Yields
        ------
        self.rawdata_dict['measurement_time']['var_data'] : ndarray
            Updated ....
        
        Notes
        -----
        .. TODO :: Finish docstring.
        """

        logging.info('date consistency-check... ')
        if self.mdate_consistency() == False:
            logging.info('date in nc-file will be replaced with date of filename.')
            np_array = np.array(self.rawdata_dict['measurement_time']['var_data']) ## converting to numpy-array for easier numpy-operations
            mdate = self.mdate_filename()
            np_array[:, 0] = mdate ## assign new date value to the whole first column of the 2d-numpy-array
            self.rawdata_dict['measurement_time']['var_data'] = np_array


    def setChannelTags(self):
        """Set the channel tags.

        they are stored as dictionary in
        ```
        data_cube.channel_dict
        ```

        and as list in 
        ```
        data_cube.retrievals_highres['channel']
        data_cube.polly_config_dict['channelTags']
        ```

        as an array of boolean flags in
        ```
        data_cube.flags
        ```

        as flags per channel in 
        ```
        data_cube.flag_355_total_FR
        ```

        Yields
        ------
        self.retrievals_highres['channel'] : list
            Updated channel tags.
        self.polly_config_dict['channelTags'] : list
            updated channel tags.
        self.flag_355_total_FR : ndarray
            True if the channel is `355nm total FR` else False. 
        self.flag_355_cross_FR : ndarray
            True if the channel is `355nm cross FR` else False. 
        self.flag_355_parallel_FR : ndarray
            True if the channel is `355nm parallel FR` else False. 
        self.flag_355_total_NR : ndarray
            True if the channel is `355nm total NR` else False. 
        self.flag_387_total_FR : ndarray
            True if the channel is `387nm total FR` else False. 
        self.flag_387_total_NR : ndarray
            True if the channel is `387nm total NR` else False. 
        self.flag_407_total_FR : ndarray
            True if the channel is `407nm total FR` else False. 
        self.flag_407_total_NR : ndarray
            True if the channel is `407nm total NR` else False. 
        self.flag_532_total_FR : ndarray
            True if the channel is `532nm total FR` else False. 
        self.flag_532_cross_FR : ndarray
            True if the channel is `532nm cross FR` else False. 
        self.flag_532_parallel_FR : ndarray
            True if the channel is `532nm parallel FR` else False. 
        self.flag_532_total_NR : ndarray
            True if the channel is `532nm total NR` else False. 
        self.flag_532_cross_DFOV : ndarray
            True if the channel is `532nm cross DFOV` else False. 
        self.flag_532_rr_FR : ndarray
            True if the channel is `532nm rr FR` else False. 
        self.flag_607_total_FR : ndarray
            True if the channel is `607nm total FR` else False. 
        self.flag_607_total_NR : ndarray
            True if the channel is `607nm total NR` else False. 
        self.flag_1058_total_FR : ndarray
            True if the channel is `1058nm total FR` else False. 
        self.flag_1064_total_FR : ndarray
            True if the channel is `1064nm total FR` else False. 
        self.flag_1064_cross_FR : ndarray
            True if the channel is `1064nm cross FR` else False. 
        self.flag_1064_total_NR : ndarray
            True if the channel is `1064nm total NR` else False. 

        Notes
        -----
        .. TODO::
            - Several channels are still missing channelFlags.
            - We are not consistant in the naming scheam of the RR-channels.
            - Need to implement new config flags for fluorescence and high intensity channels.
        """

        ChannelTags = pollyChannelTags.pollyChannelTags( 
            self.polly_config_dict['channelTag'],   # TODO key: channelTags vs channelTag???
            flagFarRangeChannel=self.polly_config_dict['isFR'],
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
            flag1064nmChannel=self.polly_config_dict['is1064nm'],
            flagDFOVChannel=self.polly_config_dict['isDFOV'],
        )

        ChannelTags, self.polly_config_dict = pollyChannelTags.polly_config_channel_corrections(chTagsOut_ls=ChannelTags, polly_config_dict=self.polly_config_dict)

        self.retrievals_highres['channel'] = ChannelTags
        self.polly_config_dict['channelTags'] = ChannelTags
        self.channel_dict = {i: item for i, item in enumerate(ChannelTags)}

        ChannelFlags = pollyChannelTags.pollyChannelflags(
            channel_dict_length=len(self.channel_dict),
            flagFarRangeChannel=self.polly_config_dict['isFR'],
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
            flag1064nmChannel=self.polly_config_dict['is1064nm'],
            flagDFOVChannel=self.polly_config_dict['isDFOV'],
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


    def preprocessing(self, collect_debug:bool=False):
        """Preprocessing of Lidar data. Includes the followin processes in order:
            1. Deadtime correction
            2. Background correction
            3. First-bin shift
            4. Mask for low-SNR
            5. Mask bins with laser shutter on
            6. Mask bins with fog
            7. Mask for depolarization-calibration process
            8. Range correction.
        
        Parameters
        ----------
        collect_debug : bool, optional
            If true, collects debug information. Default is False.
        
        Yelds
        -----
        self.retrievals_highres : dict
            With preprocessed lidar data.

            Keys
            ----
            mShots : ndarray
                Number of the laser shots for each profile.
            sigDTCor : ndarray
                Dead time corrected signal [photon count].
            BG : ndarray
                Background.
            sigBGCor : ndarray
                Backgound corrected signal [photon count].
            range : ndarray
                Height above ground at zenith angle [m].
            height : ndarray
                Height above ground [m].
            alt : ndarray
                Altitude [m].
            time : list
                Measurement time in unix time.
            time64 : ndarray
                Measurement time in numpy.datetime.
            SNR : ndarray
                Signal to noise ratio.
            lowSNRMask : ndarray
                True if SNR is less SNRmin. Otherwise False.
            shutterOnMask : ndarray
                True if at timesteps where the shutters was on. Otherwise False.
            fogMask : ndarray
                If it is foggy which means the signal will be very weak, 
                fogMask will be set True. Otherwise, False.
            mask607Off : ndarray
                Mask of PMT on/off status at 607 nm channel.
            mask387Off : ndarray
                Mask of PMT on/off status at 387 nm channel.
            mask407Off : ndarray
                Mask of PMT on/off status at 407 nm channel.
            mask355RROff : ndarray
                Mask of PMT on/off status at 355 nm rotational Raman channel.
            mask532RROff : ndarray
                Mask of PMT on/off status at 532 nm rotational Raman channel.
            mask1064RROff : ndarray
                Mask of PMT on/off status at 1064 nm rotational Raman channel.
            depCal_P_Ang_time_start : list
                Time for the first profile with valid positive angle depolarization 
                calibration.
            depCal_P_Ang_time_end : list
                Time for the last profile with valid positive angle depolarization 
                calibration.
            depCal_N_Ang_time_start : list
                Time for the first profile with valid negative angle depolarization 
                calibration.
            depCal_N_Ang_time_end : list
                Time for the last profile with valid negative angle depolarization
                calibration.
            maskDepCal : ndarray
                If polly was doing polarization calibration, depCalMask is set
                True. Otherwise, False.
            RCS : ndarray
                Range Corrected signal [MCPS].
            PCR_slice : ndarray
                Background corrected signal in photon count rate [MCPS].
        
        Notes
        -----
        .. TODO::
            - This is just a first draft for a docstring. Improve it. There is more processes and outputs of the function. 
            - Also go over the inputs and see if all of them are actually needed.     
        """

        logging.info("Preprocessing ...")
        preproc_dict = pollyPreprocess.pollyPreprocess(
            self.rawdata_dict,
            deltaT=self.polly_config_dict['deltaT'],
            flagForceMeasTime=self.polly_config_dict['flagForceMeasTime'],
            maxHeightBin=self.polly_config_dict['max_height_bin'],
            firstBinIndex=self.polly_config_dict['first_range_gate_indx'],
            firstBinHeight=self.polly_config_dict['first_range_gate_height'],
            pollyType=self.polly_config_dict['name'],
            flagDeadTimeCorrection=self.polly_config_dict['flagDTCor'],
            deadtimeCorrectionMode=self.polly_config_dict['dtCorMode'],
            deadtimeParams=self.polly_config_dict['dt'],
            flagSigTempCor=self.polly_config_dict['flagSigTempCor'],
            tempCorFunc=self.polly_config_dict['tempCorFunc'],
            meteorDataSource=self.polly_config_dict['meteorDataSource'],
            gdas1Site=self.polly_config_dict['gdas1Site'],
            gdas1_folder=self.picasso_config_dict['gdas1_folder'],
            radiosondeSitenum=self.polly_config_dict['radiosondeSitenum'],
            radiosondeFolder=self.polly_config_dict['radiosondeFolder'],
            radiosondeType=self.polly_config_dict['radiosondeType'],
            bgCorrectionIndexLow=self.polly_config_dict['bgCorRangeIndxLow'],
            bgCorrectionIndexHigh=self.polly_config_dict['bgCorRangeIndxHigh'],
            asl=self.polly_config_dict['asl'],
            initialPolAngle=self.polly_config_dict['init_depAng'],
            maskPolCalAngle=self.polly_config_dict['maskDepCalAng'],
            minSNRThresh=self.polly_config_dict['mask_SNRmin'],
            minPC_fog=self.polly_config_dict['minPC_fog'],
            flagFarRangeChannel=self.polly_config_dict['isFR'],
            flag532nmChannel=self.polly_config_dict['is532nm'],
            flagTotalChannel=self.polly_config_dict['isTot'],
            flag355nmChannel=self.polly_config_dict['is355nm'],
            flag607nmChannel=self.polly_config_dict['is607nm'],
            flag387nmChannel=self.polly_config_dict['is387nm'],
            flag407nmChannel=self.polly_config_dict['is407nm'],
            flag355nmRotRaman=np.bitwise_and(np.array(self.polly_config_dict['is355nm']), np.array(self.polly_config_dict['isRR'])).tolist(), # TODO: is this not rdundent as we already should have an RR mask
            flag532nmRotRaman=np.bitwise_and(np.array(self.polly_config_dict['is532nm']), np.array(self.polly_config_dict['isRR'])).tolist(), # TODO: is this not rdundent as we already should have an RR mask
            flag1064nmRotRaman=np.bitwise_and(np.array(self.polly_config_dict['is1064nm']), np.array(self.polly_config_dict['isRR'])).tolist(), # TODO: is this not rdundent as we already should have an RR mask. Also this will not work for the 1058nm channel.
            isUseLatestGDAS=self.polly_config_dict['flagUseLatestGDAS'],
            collect_debug=collect_debug,
            flagPicassoComparison=self.polly_config_dict['flagPicassoComparison'],
        )
        self.retrievals_highres.update(preproc_dict)


    def SaturationDetect(self):
        """Saturation Detection.
        
        Yields
        ------
        self.flagSaturation : ndarray
            1 dimensional temporal boolean array per channel.
            True if the channel is saturated, else False.
        """

        logging.info("Saturation detection ...")
        self.flagSaturation = pollySaturationDetect.pollySaturationDetect(
            data_cube = self,
            sigSaturateThresh = self.polly_config_dict['saturate_thresh']
        )


    def polarizationCaliD90(self, db_path:str=None):
        """Calibration with the Delta-90 method.

        Parameters
        ----------
        db_path : str
            Path to database to read DC values from in case none
            are successfully retrieved. Default is None.
        
        Yields
        ------
        self.pol_cali['D90' & 'D90_db'] : dict
            All retrieved and read delta-90 depol calibration constants and retrieval information.
        self.etaused : dict
            The used depol calibration constants per channel.
        
        Notes
        -----
        - The stuff that starts here in the matlab version
          https://github.com/PollyNET/Pollynet_Processing_Chain/blob/5efd7d35596c67ef8672f5948e47d1f9d46ab867/lib/interface/picassoProcV3.m#L442
        """

        logging.info("Delta 90 polarization calibration ...")
        polarization.loadGHK(self)
        self.pol_cali['D90'] = polarization.calibrateGHK(self)
        isUsable = [element['status'] for key, val in self.pol_cali['D90'].items() for element in val]

        if np.sum(isUsable) > 0:
            logging.info("Using retieved polarization calibration constants.")
            self.etaused = select.single_best(self.pol_cali['D90'], 'eta', 'eta_std')
        elif db_path is not None:
            logging.warning("Can not retieve viable polarization calibration constants, uses constants form the database.")
            table_name = 'depol_calibration_constant'
            ts_interval = self.retrievals_highres['time'][0], self.retrievals_highres['time'][-1]
            self.pol_cali['D90_db'] = sql_db.get_from_sql_db(db_path, table_name, ts_interval)['D90_db']
            self.etaused = select.single_best(self.pol_cali['D90_db'], 'eta', 'eta_std')
        else:
            logging.critical("Can not retieve viable polarization calibration constants, and no database detected.")
            raise ValueError("Can not retieve viable polarization calibration constants, and no database detected.")


    def cloudScreen(self, collect_debug:bool=False):
        """Basic cloud screening.

        Parameters
        ----------
        collect_debug : bool, optional
            If true, collects debug information. Default is False.
        
        Yields
        ------
        self.flagCloudFree : ndarray
            1 dimensional temporal boolean array. 0 = cloudy, 1 = cloud free. 
        
        Notes
        -----
        - Matlab equivalent code at:
          https://github.com/PollyNET/Pollynet_Processing_Chain/blob/b3b8ec7726b75d9db6287dcba29459587ca34491/lib/interface/picassoProcV3.m#L663
        """
        
        logging.info("Cloud screening ...")
        self.flagCloudFree = cloudscreen.cloudscreen(self, collect_debug=collect_debug)


    def cloudFreeSeg(self):
        """Cloud free profile segmentation.

        Yields
        ------
        self.clFreeGrps : nested list
            List with start, stop index of each cloud free segement.
        
        Notes
        -----
        - The matlab eqvivalent code starts here:
          https://github.com/PollyNET/Pollynet_Processing_Chain/blob/b3b8ec7726b75d9db6287dcba29459587ca34491/lib/interface/picassoProcV3.m#L707
        .. TODO:: Write a more extensive example for the docstring.
          
        Example
        -------
        .. code-block:: python
        
            data_cube.clFreGrps = [
                [35, 300],
                [2500, 2800]
            ]
        """

        logging.info("Segment cloud free groups ...")
        self.clFreeGrps = profilesegment.segment(self)


    def aggregate_profiles(self, var:str|list=None, func=np.nansum):
        """Aggregate highres profiles over cloud free segments.

        Parameters
        ----------
        var : str or array_like
            Name of variable to aggregate. Default is None.
        func : function, optional
            Function to do the aggregation (mean, sum, median, etc.). Default is np.nansum.
        
        Yields
        ------
        self.retrievals_profiles[`var`] : ndarray
            Aggregated profile.
        
        Notes
        -----
        - The variable(s) `var` need to be stored in the dictionary data_cube.retrievals_highres.
          All aggregated profiles will be stored in the dictionary data_cube.retrievals_profiles under the
          the same key.
        - If var is None. The following variables will be aggregated:
          `sigBGCor`, `BG`, `RCS`, `mShots`, `mask387Off`, `mask607Off`, `mask407Off`
        - `func=np.nansum` is needed for correct SNR calculations.

        ** History **

        - xxxx-xx-xx: First edition by ...
        - 2026-07-28: Added option to aggregate multiple variables in the same call.
                      And corrected the aggregation for PCR-signals like `RCS`.
        
        """

        if var is None:
            # Take care of the default scenario
            self.aggregate_profiles(['sigBGCor', 'BG', 'RCS', 'mShots'])
            self.aggregate_profiles(['mask387Off', 'mask607Off', 'mask407Off'], np.nanmean)
            return

        if isinstance(var, str):
            var = [var]

        for variable in var:
            logging.info(f"Aggregating variable: {variable} ...")
            if variable in self.retrievals_highres:
                if variable == "RCS":
                    if self.polly_config_dict['flagPicassoComparison']:
                        self.retrievals_profile[variable] = \
                            preprocprofiles.aggregate_clFreeGrps(self, variable, np.nanmean)
                    else:
                        self.retrievals_profile[variable] = pollyPreprocess.calculate_rcs(
                            signal=pollyPreprocess.photonCount2PCR(
                                signal=preprocprofiles.aggregate_clFreeGrps(self, 'sigBGCor', func),
                                mShots=preprocprofiles.aggregate_clFreeGrps(self, 'mShots', func),
                                hRes=self.rawdata_dict['measurement_height_resolution']['var_data']
                            ),
                            ranges=self.retrievals_highres['range']
                        )
                else:
                    self.retrievals_profile[variable] = \
                        preprocprofiles.aggregate_clFreeGrps(self, variable, func)
            else:
                logging.critical(f"{variable} is NOT in data_cube.retrievals_highres")
                raise ValueError(f"Could not locate variable '{variable}'.")


    def loadMeteo(self):
        """Load meteorological data.
        
        Meteorological data like;
        - Temperatue
        - Pressure
        - Relative Humidity
        - Spesific Humidity

        is read form the cloudnet ECMWF file specified by the config variable;
        - `meteorDataSource`
        - `meteo_folder`
        - `meteo_file`

        and saved in the dataFrame `self.met`.
        
        Yields
        ------
        self.met : object
            Object to handle meterological data.

        Notes
        -----
        - Only `meteorDataSource = 'nc_cloudnet'` is currently suported.
        """

        logging.info("Loading meteorological data ...")
        self.met = readMeteo.Meteo(
            self.polly_config_dict['meteorDataSource'], 
            self.polly_config_dict['meteo_folder'],
            self.polly_config_dict['meteo_file']
        )
        self.met.load(
            times=datetime.datetime.timestamp(datetime.datetime.strptime(self.date, '%Y%m%d')),
            heights=self.retrievals_highres['height'], asl=float(self.polly_config_dict['asl']),
            flagPicassoComparison=self.polly_config_dict['flagPicassoComparison']
        )


    def loadAOD(self):
        """Load the AOD from a co-located fotometer.

        Notes
        -----
        .. TODO:: Implement the option to load AOD from co-located fotometer.
        """

        logging.critical('Function loadAOD is not yet implemented!')
        raise NotImplementedError('Function loadAOD is not yet implemented!')


    def calcMolecular(self):
        """Calculate the molecular scattering for the cloud free periods
        with the strategy of first averaging the met data and then
        calculating the rayleigh scattering.

        Yields
        ------
        self.mol_profiles : dict
            Dictionary containing calculated molecular profiles for each channel.
            
            Keys
            ----
            mBsc_355 : ndarray (channels, height)
                Molecular backscatter at 355 nm [m^{-1}Sr^{-1}].
            mExt_355 : ndarray (channels, height)
                Molecular extinction at 355 nm [m^{-1}].
            mBsc_387 : ndarray (channels, height)
                Molecular backscatter at 387 nm [m^{-1}Sr^{-1}].
            mExt_387 : ndarray (channels, height)
                Molecular extinction at 387 nm [m^{-1}].
            mBsc_407 : ndarray (channels, height)
                Molecular backscatter at 407 nm [m^{-1}Sr^{-1}].
            mExt_407 : ndarray (channels, height)
                Molecular extinction at 407 nm [m^{-1}].
            mBsc_532 : ndarray (channels, height)
                Molecular backscatter at 532 nm [m^{-1}Sr^{-1}].
            mExt_532 : ndarray (channels, height)
                Molecular extinction at 532 nm [m^{-1}].
            mBsc_607 : ndarray (channels, height)
                Molecular backscatter at 607 nm [m^{-1}Sr^{-1}].
            mExt_607 : ndarray (channels, height)
                Molecular extinction at 607 nm [m^{-1}].
            mBsc_1058 : ndarray (channels, height)
                Molecular backscatter at 1058 nm [m^{-1}Sr^{-1}].
            mExt_1058 : ndarray (channels, height)
                Molecular extinction at 1058 nm [m^{-1}].
            mBsc_1064 : ndarray (channels, height)
                Molecular backscatter at 1064 nm [m^{-1}Sr^{-1}].
            mExt_1064 : ndarray (channels, height)
                Molecular extinction at 1064 nm [m^{-1}].
            number_density : ndarray (channels, height)
                Number density.
        
        Notes
        -----
        .. TODO:: How are these molecular profiles aggregated (mean or sum??).
        .. TODO:: Idea: As we already need the highres molecular signal for the WV-product. Would it not make more sens to
                  retrieve this first and then use the aggregate_profile() function to get the cloud free profiles?
                  --> This only makes sens if we can directly retrieve the highres molecular signal which I dout we can.
        """

        logging.info("Calculating molecular profiles ...")
        time_slices = [self.retrievals_highres['time64'][grp] for grp in self.clFreeGrps]
        logging.info(f"time slices of cloud free {time_slices}")
        mean_profiles = self.met.get_mean_profiles(time_slices)
        self.mol_profiles = molecular.calc_profiles(mean_profiles, flagPicassoComparison=self.polly_config_dict['flagPicassoComparison'])
    

    def rayleighFit(self, collect_debug:bool=False):
        """Perform the rayleigh fit procedure.

        Parameters
        ----------
        collect_debug : bool, optional
            If true, collects debug information. Default is False.
        
        Yields
        ------
        self.retrievals_profiles['refH'][idx][ch] : dict
            Reference height information for cloud free segment `idx` and channel `ch`.
        
            Keys
            ----
            DPInd : list
                List of Douglas-Peckur indecies used as potensial reference index candidates.
            refInd : list
                Indices (start, stop) of the retrieved reference height.
            refHeigt : list
                Height above ground (start, stop) of the retrieved reference height.
            refRange : list
                Height above ground at zenith angle (start, stop) of the retrieved reference height.

        Notes
        -----
        - Direct translation from the matlab code. There might be noticeable numerical discrepancies (especially in the residual)
          seemed to work ok for 532, 1064, but with issues for 355.
        - The Douglas Peucker algorithm works fine (gives the same result as the Matlab version). However, due to the numerical discrepancies
          In the residuals of the fit algorithm sometimes different refH segments are chosen.
        """

        logging.info('Start Rayleigh Fit')
        logging.warning(f'Potential for differences to matlab code due to numerical issues (subtraction of two small values)')

        self.retrievals_profile['refH'] = rayleighfit.rayleighfit(self, collect_debug)


    def polarizationCaliMol(self):
        """Calibration with molecular signal in reference height.
        
        Yields
        ------
        self.pol_cali['mol'][ch][idx] : dict
            Molecular depol calibration constants and retrieval information for 
            channel `ch` and cloud free index `idx`.

            keys
            ----
            eta : ndarray
                Polarization calibration eta.
            etaStd : ndarray
                Uncertainty of polarization calibration eta.
            fac : array
                Polarization calibration factor.
            facStd : ndarray
                Uncertainty of polarization calibration factor.
            status : int
                Retrieval status
                    0 : Bad
                    1 : Good
            time_start : int
                Start time of the cloud free segment for the retrieval in unixtime.
            end_time : int
                End time of the cloud free segment for the retrieval in unixtime.

        Notes
        -----
        - Calibration is only performed if the config variable `flagMolDepolCali = True`.
        """

        if self.polly_config_dict['flagMolDepolCali']:
            logging.info("Calibrating molecular signal ...")
            logging.warning(f'not checked against the matlab code')
            self.pol_cali['mol'] = polarization.calibrateMol(self)
        else:
            logging.warning("'flagMolDepolCali' set to False")


    def transCor(self):
        """Perform GHK-transmission correction on the signal.

        Yields
        ------
        self.retrievals_highres : dict
            With corrected signal.

            Keys
            ----
            sigTCor : ndarray
                GHK-Transmission corrected signal [MCPS].
            BGTCor : ndarray
                GHK-Transmission corrected background.

        Notes
        -----
        - if the config entry `flagTransCor = False`, no GHK transmission correction
          will be done and the background corrected signal will replace the transmission
          corrected signal for the rest of the processing steps.
        
        .. TODO:: Overwriting `sigTCor` with `sigBGCor` when `flagTransCor = False` can
                  cause confiusion among useres. It would be more intutive to not have
                  a `sigTCor` variable in this case. However, this requiers re... the
                  rest of the processing chain.
        """

        if self.polly_config_dict['flagTransCor']:
            logging.info('GHK Transmission correction ...')
            self.retrievals_highres['sigTCor'], self.retrievals_highres['BGTCor'] = \
                transCor.transCorGHK_cube(self)
        else:
            logging.warning('No GHK transmission correction done.')
            self.retrievals_highres['sigTCor'], self.retrievals_highres['BGTCor'] = \
                self.retrievals_highres['sigBGCor'], self.retrievals_highres['BG']


    def retrievalKlett(self, oc:bool=False, nr:bool=False):
        """Apply Klett retrieval.

        Parameters
        ----------
        oc : bool, optional
            If true, apply the retrieval on the overlap corrected profiles. The outupt will then also be saved under
            the key 'klett_OC' insted of 'klett'. Default is False.
        nr : bool, optional
            If true, apply the retrieval on the near range profiles as well as the far range profiles. Default is False.
        
        Yields
        ------
        self.retrievals_profile['klett' or 'klett_OC'][idx][ch] : dict
            Klett retrieved optical profiles and retrieval information
            for cloud free segment `idx` and channel `ch`.

            Keys
            ----
            aerExt : ndarray
                Aerosol extinction coefficient [m^{-1}].
            aerExtStd : ndarray
                Uncertainty of aerosol extinction coefficient [m^{-1}].
            aerBsc : ndarray
                Aerosol backscatter coefficient [m^{-1}Sr^{-1}].
            aerBscStd : ndarray
                Uncertainty of aerosol backscatter coefficient [m^{-1}Sr^{-1}].
            aerBR : ndarray
                Aerosol backscatter ratio.
            aerBRStd : ndarray
                Statistical uncertainty of aerosol backscatter ratio.
            retrieval : str
                Name of retrieval type, eg. 'klett'.
            signal : str
                Name of the signal used for the retrieval, eg. 'TCor'.
            refBeta : float
                Reference value used for the retrieval [m^{-1}Sr^{-1}].
        
        Notes
        -----
        - The retrieval is by default applied on the FR profiles. if `nr=True` the retrieval will be applied both on
          the NR and FR profiles.       
        """

        retrievalname = 'klett'
        kwargs = {}
        if oc:
            # Check if overlap corrected signal exist
            if "sigOLCor" not in self.retrievals_highres:   # This should be done more general not only for OLCor but for other signals as well.
                logging.warning("No overlap corrected signal found.")
                return
            retrievalname +='_OC'
            kwargs['signal'] = 'OLCor'
        if nr:
            kwargs['nr'] = True

        logging.info(f"Klett retrieval for FR {'& NR' if nr else None} {'overlap' if oc else 'GHK-transmission'} corrected signal ...")
        self.retrievals_profile[retrievalname] = klettfernald.run_cldFreeGrps(self, **kwargs)
        if retrievalname not in self.retrievals_profile['avail_optical_profiles']:
            self.retrievals_profile['avail_optical_profiles'].append(retrievalname)


    def retrievalRaman(self, oc:bool=False, nr:bool=False, collect_debug:bool=False):
        """Apply raman retrieval (nighttime only).
        
        Parameters
        ----------
        oc : bool, optional
            If true, apply the retrieval on the overlap corrected profiles. The outupt will then also be saved under
            the key 'raman_OC' insted of 'raman'. Default is False.
        nr : bool, optional
            If true, apply the retrieval on the near range profiles as well as the far range profiles. Default is False.
        collect_debug : bool, optional
            If true, collects debug information. Default is False.
        
        Yields
        ------
        self.retrievals_profile['raman' or 'raman_OC'][idx][ch] : dict
            Raman retrieved optical profiles and retrieval information
            for cloud free segment `idx` and channel `ch`.

            Keys
            ----
            aerExt : ndarray
                Aerosol extinction coefficient [m^{-1}].
            aerExtStd : ndarray
                Uncertainty of aerosol extinction coefficient [m^{-1}].
            aerBsc : ndarray
                Aerosol backscatter coefficient [m^{-1}Sr^{-1}].
            aerBscStd : ndarray
                Uncertainty of aerosol backscatter coefficient [m^{-1}Sr^{-1}].
            LR : ndarray
                Aerosol Lidar ratio [sr].
            effRes : ndarray
                Effective resolution of aerosol lidar ratio [m].
            LRStd : ndarray
                Uncertainty of aerosol lidar ratio [sr].
            retrieval : str
                Name of retrieval type eg. 'raman'.
            signal : str
                Name of the signal used for the retrieval eg. 'TCor'.
            refBeta : float
                Reference value used for the retrieval.

        Notes
        -----
        - The retrieval is by default applied on the FR profiles. if `nr=True` the retrieval will be applied both on
          the NR and FR profiles.
        """

        retrievalname = 'raman'
        kwargs = {}
        if oc:
            # Check if overlap corrected signal exist
            if "sigOLCor" not in self.retrievals_highres:   # This should be done more general not only for OLCor but for other signals as well.
                logging.warning("No overlap corrected signal found.")
                return
            retrievalname +='_OC'
            kwargs['signal'] = 'OLCor'

            # get the full overlap height for the overlap corrected variant
            # group by the cloud free groups 
            kwargs['heightFullOverlap'] = \
                [np.mean(self.retrievals_highres['heightFullOverCor'][slice(*cF)], axis=0) for 
                 cF in self.clFreeGrps]
        if nr:
            kwargs['nr'] = True
        kwargs['collect_debug'] = collect_debug

        logging.info(f"Raman retrieval for FR {'& NR' if nr else None} {'overlap' if oc else 'GHK-transmission'} corrected signal ...")
        self.retrievals_profile[retrievalname] = raman.run_cldFreeGrps(self, **kwargs)
        if retrievalname not in self.retrievals_profile['avail_optical_profiles']:
            self.retrievals_profile['avail_optical_profiles'].append(retrievalname)


    def overlapCalc(self, collect_debug:bool=False):
        """Estimate the overlap function.

        Parameters
        ----------
        collect_debug : str, optional
            If true, collects debug information. Default is False.
        
        Yields
        ------
        self.retrievals_profile['overlap']['frnr'][idx][ch] : list of dicts
            Far-range-Near-range retrieved overlap function and retrieval
            information for cloud free segment `idx` and channel `ch`.
            
            Keys
            ----
            olFunc : ndarray
                Overlap function.
            olFuncStd : ndarray
                Standard deviation of overlap function.
            sigRatio : float
                Signal ratio between near-range and far-range signals.
            normRange : list
                Height index of the signal normalization range.

        self.retrievals_profile['overlap']['raman'][idx][ch] : list of dicts
            Raman retrieved overlap function and retrieval information
            for cloud free segment `idx` and channel `ch`.

            Keys
            ----
            olFunc : ndarray
                Overlap function.
            olFunc_raw : ndarray
                Overlap function with no smoothing.
            LR : float
                Optimal Lidar Ratio for ...
            normRange : list
                Height index of the signal normalization range.
        
        Notes
        -----
        - Different to the matlab version, where an average over all cloud
          free periods is taken. Here the overlap is applied per cloud free period.
        """

        if not self.polly_config_dict["flagOLCor"]:
            return
        logging.info("calculating overlap functions ...")
        self.retrievals_profile['overlap'] = {}
        self.retrievals_profile['overlap']['frnr'] = overlapEst.run_frnr_cldFreeGrps(self, collect_debug=collect_debug)
        self.retrievals_profile['overlap']['raman'] = overlapEst.run_raman_cldFreeGrps(self, collect_debug=collect_debug)


    def overlapFixLowestBins(self):
        """The lowest bins are affected by stange near range effects.

        Yields
        ------
        self.retrievals_profile['overlap']['frnr'][idx][ch]['olFunc'] : ndarray
            FRNR retrieved overlap func with subdued near-range effect.
        self.retrievals_profile['overlap']['raman'][idx][ch]['olFunc'] : ndarray
            Raman retrieved overlap func with subdued near-range effect.
        """

        if not self.polly_config_dict["flagOLCor"]:
            return
        height = self.retrievals_highres['range']
        for k in self.retrievals_profile['overlap']:
            logging.info(f"Fixing lower bins for {k} overlap functions")
            overlapCor.fixLowest(self.retrievals_profile['overlap'][k], np.where(height > 800)[0][0])


    def overlapCor(self):
        """Overlap correction.

        Yields
        ------
        self.retrieval_highres : dict
            With overlap corrected signal and retrieval information.

            Keys
            ----
            overlap2d : ndarray
                2 dimensional (time, height) overlap function.
            sigOLCor : ndarray
                Overlap corrected signal [photon count].
            BGOLCor : ndarray
                Overlap corrected background.
            heightFullOverCor : ndarray
                Retrieved height of full overlap [m].
        
        Notes
        -----
        - the overlap correction is implemented differently to the matlab version
          first a 2d (time, height) correction array is constructed then it is applied.
          In future this will allow for time varing overlap functions
        """

        if self.polly_config_dict['overlapCorMode'] == 0 or not self.polly_config_dict["flagOLCor"]:
            logging.info('no overlap correction applied')
            return
        logging.info('Apply overlap correction ...')
        if self.polly_config_dict['overlapCorMode'] == 1:
            logging.info('overlapCorMode 1 -> need file for overlapfunction')
            if not 'overlap' in self.retrievals_profile:
                self.retrievals_profile['overlap'] = {}
            self.retrievals_profile['overlap']['file'] = overlapEst.load(self)
        self.retrievals_highres['overlap2d'] = overlapCor.spread(self)
        ret = overlapCor.apply_cube(self)
        self.retrievals_highres['sigOLCor'] = ret[0]
        self.retrievals_highres['BGOLCor'] = ret[1]
        self.retrievals_highres['heightFullOverCor'] = ret[2]


    def calcDepol(self):
        """Calculate the volume depol and the particle depol.
        
        Yields
        ------
        self.retrievals_profiles[sig][idx][ch] : dict
            Volume and particle depolarization profiles and retrieval info
            for signal type `sig` for cloud free segment `idx` and channel `ch`.

            Keys
            ----
            vdr : ndarray
                Volume depolarization ratio [].
            vdrStd : ndarray
                Uncertainty of the volume depolarization ratio.
            mdr : ndarray
                Molecular depolarization ratio [].
            mdrStd : ndarray
                Uncertainty of the molecular depolarization ratio.
            pdr : ndarray
                Particle depolarization ratio [].
            pdrStd : ndarray
                Uncertainty of the particle depolarization ratio.

        
        Notes
        -----
        - This retrieval is done for all avaiable optical profiles.
        """

        for ret_prof_name in self.retrievals_profile['avail_optical_profiles']:
            logging.info(f"Calculate volume and particle depolarization ratios for product {ret_prof_name} ...")
        
            self.retrievals_profile[ret_prof_name] = depolarization.voldepol_cldFreeGrps(
                self, ret_prof_name) 
            self.retrievals_profile[ret_prof_name] = depolarization.pardepol_cldFreeGrps(
                self, ret_prof_name) 


    def estQualityMask(self):
        """Estimate the quality mask.
        
        Yields
        ------
        self.retrievals_highres['quality_mask'] : ndarray
            High resolution (time, height) quality masks per channel.
                0 : good data
                1 : low-SNR data
                2 : depolarization calibration periods
                3 : shutter on
                4 : fog
                5 : saturated (NEW)
        
        Notes
        -----
        - If the config variables `flagUseImprovedSNR=True`, SNR and the lowSNRMask are
          recalculated with quasi-smoothed signals to improve data quality.

        ** History **
        
        - xxxx-xx-xx: First edition by ...
        - 2026-07-17: Added quasi-smoothed SNR and lowSNRMask.

        """

        logging.info("Estimate quality masks ...")
        if self.polly_config_dict['flagUseImprovedSNR']:
            self.retrievals_highres['SNR_quasi'], self.retrievals_highres['lowSNRMask_quasi'] = qualityMask.improvedSNR(self)

        self.retrievals_highres['quality_mask'] = qualityMask.qualityMask(self)


    def Angstroem(self):
        """Calculate the angstrom exponent.

        Yields
        ------
        self.retrievals_profiles[sig][idx][ch] : dict
            Volume and particle depolarization profiles and retrieval info
            for signal type `sig` for cloud free segment `idx` and channel `ch`.

            Keys
            ----
            AE_{prod}_{wv1}_{wv2} : ndarray
                Angstroem Exponent for product `prod`, and wavelengths `wv1` and `wv2`.
            AEStd_{prod}_{wv1}_{wv2} : ndarray
                Uncertainty of Angstroem Exponent for product `prod`, and wavelengths `wv1` and `wv2`.
        
        Notes
        -----
        - AE_{prod}_{wv1}_{wv2} = log(prod_wv1 / prod_wv2) / log(wv2 / wv1)
        - This retrieval is done for all avaiable optical profiles.
        """

        for ret_prof_name in self.retrievals_profile['avail_optical_profiles']:
            logging.info(f"Calculate Angstrom exponents for product {ret_prof_name} ...")
        
            self.retrievals_profile[ret_prof_name] = angstroem.ae_cldFreeGrps(
                self, ret_prof_name) 


    def LidarCalibration(self, db_path:str=None, collect_debug:bool=False):
        """Calculate the lidar constant.

        Yields
        ------
        self.LC['klett' & 'klett_db'] : dict
            All retrieved and read Klett lidar calibration constants and retrieval information.
        self.LC['raman' & 'raman_db'] : dict
            All retrieved and read Raman lidar calibration constants and retrieval information.
        self.LCused : dict
            The lidar claibration constant used per channel.

        Notes
        -----
        .. TODO:: Find out how we prioritise raman, klett, and database retrieved LC...
        """

        logging.info("Calculating lidar calibration constants ...")
        self.LC['klett'] = lidarconstant.lc_for_cldFreeGrps(
            self, 
            retrieval='klett', 
            collect_debug=collect_debug
        )
        self.LC['raman'] = lidarconstant.lc_for_cldFreeGrps(
            self, 
            retrieval='raman', 
            collect_debug=collect_debug
        )

        logging.info("Choosing best LC per channel...")
        if db_path is None:
            logging.info("No database path found. Using retrieved LC values.")
            self.LC['klett_db'] = {}
            self.LC['raman_db'] = {}
        else:
            logging.info("Database LC values will be used when no retrieved ones are available.")
            # db_path = self.polly_config_dict['calibrationDB']
            table_name = 'lidar_calibration_constant'
            ts_interval = self.retrievals_highres['time'][0], self.retrievals_highres['time'][-1]
            self.LC['klett_db'] = sql_db.get_from_sql_db(db_path, table_name, ts_interval)['klett_db']
            self.LC['raman_db'] = sql_db.get_from_sql_db(db_path, table_name, ts_interval)['raman_db']
        
        # Prioritise Raman retrieved LCs but use Klett retrieved ones when no Raman retrieval exists.
        self.LCused =  select.single_best(self.LC['klett_db'], 'LC', 'LCStd', relative=True) |\
                         select.single_best(self.LC['raman_db'], 'LC', 'LCStd', relative=True) |\
                         select.single_best(self.LC['klett'], 'LC', 'LCStd', relative=True) |\
                         select.single_best(self.LC['raman'], 'LC', 'LCStd', relative=True)


    def attBsc_volDepol(self):
        """Highres attBsc and voldepol in 2d.
        
        Yields
        ------
        self.retrievals_highres : dict
            With attenuated backscatter.

            Keys
            ----
            attBsc_{wv}\_{t}\_{tel} : ndarray
                High resolution (time, height) attenuated backscatter at chennel {wv}\_{t}\_{tel}.
            attBsc_{wv}\_{t}\_OC : ndarray
                High resolution (time, height) overlap corrected attenuated backscatter at channel {wv}\_{t}\_FR.
        """

        # for now try with mutable state in data_cube
        logging.info("2D attenuated backscatter retrieval ...")
        highres.attbsc_2d(self)

        logging.info("2D volume depolarization ratio retrieval ...")
        highres.voldepol_2d(self)


    def molecularHighres(self):
        """Calculate the molecular signal for the 2d high resolution.
        
        Yields
        ------
        self.mol_2d : xr.Dataset
            xarray dataset with dimensions time and height and variables molecular
            backscatter (mBsc) and molecular extinction (mExt) per wavelength.

        Notes
        -----
        - mol_2d's time dimension differe from the time dimension of the measurement.
        """

        logging.info("2D molecular retrieval ...")
        self.mol_2d = molecular.calc_2d(
            self.met.ds,
            flagPicassoComparison=self.polly_config_dict['flagPicassoComparison']
        )


    def quasiV1(self):
        """QuasiV1 retrivals and target categorisation.
        
        Yields
        ------
        self.retrieval_highres : dict
            With high resolution Quasi version 1 products as well as Target categorization.

            Keys
            ----
            quasiBscV1_{wv}\_{t}\_{tel} : ndarray
                Quasi version 1 particle backscatter at channel `{wv}_{t}_{tel}`.
            quasiExtV1_{wv}\_{t}\_{tel} : ndarray
                Quasi version 1 particle extinction at channel `{wv}_{t}_{tel}`.
            quasiVdrV1_{wv}\_{t}\_{tel} : ndarray
                Quasi version 1 volume depolarization ratio at channel `{wv}_{t}_{tel}`. (usally `wv =532`).
            quasiPdrV1_{wv}\_{t}\_{tel} : ndarray
                Quasi version 1 particle depolarization ratio at channel `{wv}_{t}_{tel}`. (usally `wv =532`).
            quasiAEV1_{wv1}\_{wv2} : ndarray
                Quasi version 1 Angstroem exponent between two wavelengets `wv1` and `wv2`.
            tcMaskV1 : ndarray
                Classification mask version 1 (time x height).
                    0: No signal
                    1: Clean atmosphere
                    2: Non-typed particles/low conc.
                    3: Aerosol: small
                    4: Aerosol: large, spherical
                    5: Aerosol: mixture, partly non-spherical
                    6: Aerosol: large, non-spherical
                    7: Cloud: non-typed
                    8: Cloud: water droplets
                    9: Cloud: likely water droplets
                    10: Cloud: ice crystals
                    11: Cloud: likely ice crystal

        Notes
        -----
        - QuasiV1 products uses Klett-retrieved products.
        """

        logging.info('Calculating Quasi V1 particle backscatter coefficient')
        quasiV1.quasi_bsc(self)

        logging.info('Calculating Quasi V1 particle Depolarization ratio')
        quasi.quasi_pdr(self, version='V1')

        logging.info('Calculating Quasi V1 Ångström Exponent')
        quasi.quasi_angstrom(self, version='V1')

        logging.info('Producing V1 target categorization')
        quasi.target_cat(self, version='V1')


    def quasiV2(self):
        """QuasiV2 retrivals and target categorisation.
        
        Yields
        ------
        self.retrieval_highres : dict
            With high resolution Quasi version 2 products as well as Target categorization.

            Keys
            ----
            quasiBscV2_{wv}\_{t}\_{tel} : ndarray
                Quasi version 2 particle backscatter at channel `{wv}_{t}_{tel}`.
            quasiExtV2_{wv}\_{t}\_{tel} : ndarray
                Quasi version 2 particle extinction at channel `{wv}_{t}_{tel}`.
            quasiVdrV2_{wv}\_{t}\_{tel} : ndarray
                Quasi version 2 volume depolarization ratio at channel `{wv}_{t}_{tel}`. (usally `wv =532`).
            quasiPdrV2_{wv}\_{t}\_{tel} : ndarray
                Quasi version 2 particle depolarization ratio at channel `{wv}_{t}_{tel}`. (usally `wv =532`).
            quasiAEV2_{wv1}\_{wv2} : ndarray
                Quasi version 2 Angstroem exponent between two wavelengets `wv1` and `wv2`.
            tcMaskV2 : ndarray
                Classification mask version 2 (time x height).
                    0: No signal
                    1: Clean atmosphere
                    2: Non-typed particles/low conc.
                    3: Aerosol: small
                    4: Aerosol: large, spherical
                    5: Aerosol: mixture, partly non-spherical
                    6: Aerosol: large, non-spherical
                    7: Cloud: non-typed
                    8: Cloud: water droplets
                    9: Cloud: likely water droplets
                    10: Cloud: ice crystals
                    11: Cloud: likely ice crystal

        Notes
        -----
        - QuasiV2 products uses Raman-retrieved products.
        """

        logging.info('Calculating Quasi V2 particle backscatter coefficient')
        quasiV2.quasi_bsc(self)

        logging.info('Calculating Quasi V2 particle Depolarization ratio')
        quasi.quasi_pdr(self, version='V2')

        logging.info('Calculating Quasi V2 Ångström Exponent')
        quasi.quasi_angstrom(self, version='V2')

        logging.info('Producing V2 target categorization')
        quasi.target_cat(self, version='V2')


    def write_2_sql_db(self, parameter:str, db_path:str|None=None, method:str|None=None):
        """Write LC or eta to sqlite db table.

        Parameters
        ----------
        parameter : str
            can be LC (Lidar-calibration-constant) or DC (Depol-calibration-constant)
        method : str or NoneType
            'raman' or 'klett'
        db_path : str or NoneType
            location of the sqlite db-file

        Notes
        -----
        - The unique columns are needed that new entries overwrite old ones, 
          otherwise they are just added to the table with same timestamps.
        """

        if db_path == None:
            db_path = self.polly_config_dict['calibrationDB']
            logging.info(f"read db_path from polly_config_dict {db_path}")
        
        if parameter == 'LC':
            table_name = 'lidar_calibration_constant'
            column_names = [
                'cali_start_time', 'cali_stop_time', 'liconst', 'uncertainty_liconst', 'used_for_processing', 
                'wavelength', 'nc_zip_file', 'polly_type', 'cali_method', 'telescope']
            data_types = ['text', 'text', 'real', 'real', 'integer', 'text', 'text', 'text', 'text', 'text']
            unique=', UNIQUE(cali_start_time, cali_stop_time, wavelength, polly_type, telescope, cali_method)'
        elif parameter == 'DC':
            table_name = 'depol_calibration_constant'
            column_names = [
                'cali_start_time', 'cali_stop_time', 'depol_const', 'uncertainty_depol_const', 'used_for_processing', 
                'wavelength', 'telescope', 'nc_zip_file', 'polly_type']
            data_types = ['text', 'text', 'real', 'real', 'integer', 'text', 'text', 'text', 'text']
            unique=', UNIQUE(cali_start_time, cali_stop_time, wavelength, polly_type, telescope)'
        assert len(column_names) == len(data_types), 'column names do not match data types'

        logging.info(f'writing to sqlite-db: {db_path}')
        logging.info(f'writing {parameter} to table: {table_name}')
        
        sql_db.setup_empty(db_path, table_name, column_names, data_types, unique=unique)
        rows_to_insert = sql_db.prepare_for_sql_db_writing(self, parameter, method)

        sql_db.write_rows_to_sql_db(db_path, table_name, column_names, rows_to_insert)


    def read_calibration_db(self, db_path:str|None=None):
        """Read the calibration constants from database.

        Parameters
        ----------
        db_path : str or NoneType
            path to database file... Default is None.
        
        Yields
        ------
        self.LC : dict
            Original `LC` with all additinal lidar calibration constant availabel
            in the datafram `db_path` in the time period of the measurement +-24h.
        self.pol_cali : dict
            Original `pol_cali` with all additinal polarization calibration constant
            availabel in the datafram `db_path` in the time period of the measurement +-24h.
        
        Notes
        -----
        - Time interval includes 24h before start of the measurement and after the end of the measurement.
        """

        if db_path == None:
            db_path = self.polly_config_dict['calibrationDB']
            logging.info(f"read db_path from polly_config_dict {db_path}")

        ts_interval = self.retrievals_highres['time'][0], self.retrievals_highres['time'][-1]
        table_name = 'lidar_calibration_constant'
        self.LC.update(sql_db.get_from_sql_db(db_path, table_name, ts_interval))

        table_name = 'depol_calibration_constant'
        self.pol_cali.update(sql_db.get_from_sql_db(db_path, table_name, ts_interval))
        

    def adding_retrieving_infos_2_polly_config_dict(self):
        """Some infos from the polly_config_dict should have there own keys, e.g. reference_search_range.
        
        Yields
        ------
        self.polly_config_dict : dict
            With additional info.

            Keys
            ----
            reference_search_range_355_total_FR : list
                ...
            reference_search_range_532_total_FR : list
                ...
            reference_search_range_1064_total_FR : list
                ...
        
        Notes
        -----
        .. TODO ::
            - Finish docstring.
            - Is this function necessary?
        """
        
        lower_overlap = np.array(self.polly_config_dict['heightFullOverlap'])
        reference_search_range_355_total_FR = [int(lower_overlap[self.flag_355_total_FR][0]), self.polly_config_dict['maxDecomHeight355']]
        reference_search_range_532_total_FR = [int(lower_overlap[self.flag_532_total_FR][0]), self.polly_config_dict['maxDecomHeight532']]
        reference_search_range_1064_total_FR = [int(lower_overlap[self.flag_1064_total_FR][0]), self.polly_config_dict['maxDecomHeight1064']]
    
        self.polly_config_dict['reference_search_range_355_total_FR'] = reference_search_range_355_total_FR
        self.polly_config_dict['reference_search_range_532_total_FR'] = reference_search_range_532_total_FR
        self.polly_config_dict['reference_search_range_1064_total_FR'] = reference_search_range_1064_total_FR


#    def __str__(self):
#        return f"{self.rawdata_dict}"


    def __del__(self):
        type(self).counter -= 1

    
    def runProcessing(self):
        """Idea: Include a run function in the object as an alternative to the default run script...
        """
        raise NotImplementedError()
        

