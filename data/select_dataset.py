'''
# -----------------------------------------
Select Dataset
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    # -----------------------------------------------------------
    # CC
    # -----------------------------------------------------------
    # CC-SAG d.2.0 SC
    if dataset_type in ['cc_sag_d.2.0.complex.sc']:
        from data.dataset_CC_SAG_complex_sc_d20 import DatasetCCSAG as D

    # -----------------------------------------------------------
    # FastMRI
    # -----------------------------------------------------------
    # FastMRI d.2.1.Complex.SC
    elif dataset_type in ['fastmri.d.2.1.complex.sc']:
        from data.dataset_FastMRI_complex_sc_d21 import DatasetFastMRI as D

    # -----------------------------------------------------------
    # SKM-TEA
    # -----------------------------------------------------------
    elif dataset_type in ['skmtea.d.2.0.complex.sc']:
        from data.dataset_SKMTEA_complex_sc_d20 import DatasetSKMTEA as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
