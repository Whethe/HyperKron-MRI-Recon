'''
# -----------------------------------------
Define Training Model
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

def define_Model(opt):
    model = opt['model']

    #  2ch;
    if model in ['e2e_ir', 'e2e_ir_m16']:
        from models.model.model_e2e_ir_m16 import MRI_E2E_Recon as M

    # 1ch; ABS as GT
    elif model in ['e2e_ir_m21']:
        from models.model.model_e2e_ir_m21 import MRI_E2E_Recon as M

    # 1ch; RSS as GT
    elif model in ['e2e_ir_m22']:
        from models.model.model_e2e_ir_m22 import MRI_E2E_Recon as M

    #  2ch;
    elif model in ['e2e_gan_ir', 'e2e_gan_ir_m16']:
        from models.model.model_e2e_gan_ir_m16 import MRI_E2E_GAN_Recon as M

    # 1ch; History Version; Discarded.
    elif model in ['e2e_ir_m13']:
        raise NotImplementedError
        from models.model.backups.model_e2e_ir_m13 import MRI_E2E_Recon as M

    elif model in ['e2e_gan_ir_m13']:
        raise NotImplementedError
        from models.model.backups.model_e2e_gan_ir_m13 import MRI_E2E_GAN_Recon as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
