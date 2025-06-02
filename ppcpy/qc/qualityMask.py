
import numpy as np

def qualityMask(data_cube):
    """
    
    # 0 in quality_mask means good data
    # 1 in quality_mask means low-SNR data
    # 2 in quality_mask means depolarization calibration periods
    # 3 in quality_mask means shutter on
    # 4 in quality_mask means fog
    # 5 in quality_mask means saturated (NEW)

    The lowSNRMask is actually calculated twice, once in pollyPreprocess.m
    and then again when the quality mask is evaluated in picassoProcV3.m.
    Also the original processing chain has a quality_mask_vdr, which should be a composite
    of cross and total, maybe this can be handled more logically here.
    
    """

    print(data_cube.retrievals_highres['channel'])

    quality_mask = np.zeros_like(data_cube.retrievals_highres['sigBGCor']).astype(int)
    print('shape of quality mask', quality_mask.shape)

    for ich, ch in enumerate(data_cube.retrievals_highres['channel']):
        print(ich, ch)
        quality_mask[:, :, ich][data_cube.retrievals_highres['lowSNRMask'][:,:,ich]] = 1
        quality_mask[data_cube.retrievals_highres['depCalMask'],:,ich] = 2
        quality_mask[data_cube.retrievals_highres['shutterOnMask'],:,ich] = 3
        quality_mask[data_cube.retrievals_highres['fogMask'],:,ich] = 4
        quality_mask[:,:,ich][data_cube.flagSaturation[:,:,ich]] = 5

    return quality_mask