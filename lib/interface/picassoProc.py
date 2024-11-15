#from ..misc import *
import re
import numpy as np
import logging
import lib.misc.pollyChannelTags as pollyChannelTags
import lib.preprocess.pollyPreprocess as pollyPreprocess

class PicassoProc:
    counter = 0

    def __init__(self, rawdata_dict, polly_config_dict, picasso_config_dict):
        type(self).counter += 1
        self.rawdata_dict = rawdata_dict
        self.polly_config_dict = polly_config_dict
        self.picasso_config_dict = picasso_config_dict
        self.device = self.polly_config_dict['name']
        self.location = self.polly_config_dict['site']
        self.date = self.mdate_filename()
        self.num_of_channels = len(self.rawdata_dict['measurement_shots']['var_data'][0])
        self.data_retrievals = {}

    def mdate_filename(self):
        filename = self.rawdata_dict['filename']
        mdate = re.split(r'_',filename)[0:3]
        YYYY = mdate[0]
        MM = mdate[1]
        DD = mdate[2]
        return f"{YYYY}{MM}{DD}"

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
            logging.info('date in nc-file equals date of filename')
            return True
        else:
            logging.warning('date in nc-file differs from date of filename')
            return False

    def reset_date_infile(self):
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
        ChannelTags.remove("none")
        self.data_retrievals['channel'] = ChannelTags
        self.polly_config_dict['channelTags'] = ChannelTags
        return self

    def preprocessing(self):
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
                )
        self.data_retrievals.update(preproc_dict)

        return self

#    def __str__(self):
#        return f"{self.rawdata_dict}"

    def __del__(self):
        type(self).counter -= 1
        

