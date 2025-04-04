import os
import sys
import math
import argparse
import random
import cv2
import numpy as np
import logging
import time
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_early_stopping

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from tensorboardX import SummaryWriter
from collections import OrderedDict
from skimage.transform import resize
import lpips
import wandb


def main(json_path=''):
    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local-rank', type=int, default=None)
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # distributed settings
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # update opt
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],  net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # save opt to  a '../option.json' file
    if opt['rank'] == 0:
        option.save(opt)

    # return None for missing key
    opt = option.dict_to_nonedict(opt)

    # Do not support DDP when using WANDB Sweep
    if opt['wandb']['is_sweep']:
        assert opt['rank'] == 0, 'Do not support DDP when using WANDB Sweep'

    # configure logger
    if opt['rank'] == 0:

        # wandb init
        os.environ['WANDB_MODE'] = opt['wandb']['mode']
        wandb.init(project=opt['wandb']['project_name'],
                   entity="jiahaohuang",
                   name=opt['task'])

        # sweep parameter
        # check here when changing sweep yaml
        if opt['wandb']['is_sweep']:
            for key in wandb.config.keys():
                print(f'SWEEP PARA -- {key}: {wandb.config[key]}')
            # network
            depth = wandb.config.depth
            opt['netG']['depths'] = [depth, depth, depth, depth, depth, depth]
            num_head = wandb.config.num_head
            opt['netG']['num_heads'] = [num_head, num_head, num_head, num_head, num_head, num_head]
            opt['netG']['embed_dim'] = wandb.config.embed_dim

        # logger
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

        # tensorbordX log
        logger_tensorboard = SummaryWriter(os.path.join(opt['path']['log']))

        # wandb logger
        wandb.config.update(opt)
        wandb.define_metric("TRAIN/step")
        wandb.define_metric('TRAIN/Learning Rate', step_metric="TRAIN/step")
        wandb.define_metric('TRAIN LOSS/G_loss', step_metric="TRAIN/step")
        wandb.define_metric('TRAIN LOSS/G_loss_image', step_metric="TRAIN/step")
        wandb.define_metric('TRAIN LOSS/G_loss_frequency', step_metric="TRAIN/step")
        wandb.define_metric('TRAIN LOSS/G_loss_preceptual', step_metric="TRAIN/step")
        wandb.define_metric("VAL/step")
        wandb.define_metric('VAL LOSS/G_loss', step_metric="VAL/step")
        wandb.define_metric('VAL LOSS/G_loss_image', step_metric="VAL/step")
        wandb.define_metric('VAL LOSS/G_loss_frequency', step_metric="VAL/step")
        wandb.define_metric('VAL LOSS/G_loss_preceptual', step_metric="VAL/step")
        wandb.define_metric('VAL METRICS/SSIM', step_metric="VAL/step")
        wandb.define_metric('VAL METRICS/PSNR', step_metric="VAL/step")
        wandb.define_metric('VAL METRICS/LPIPS', step_metric="VAL/step")
        wandb.define_metric('VAL METRICS/FID', step_metric="VAL/step")

    # set seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    if 'val' not in list(opt['datasets'].keys()):
        opt['datasets']['val'] = opt['datasets']['test']
    del opt['datasets']['test']

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)

            import torch.utils.data as data
            subset_size = int(1.0 * len(train_set))
            indices = list(range(len(train_set)))
            random.shuffle(indices)
            train_set = data.Subset(train_set, indices[:subset_size])

            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=False,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=False)

        elif phase == 'val':
            val_set = define_Dataset(dataset_opt)
            val_loader = DataLoader(val_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1,
                                    drop_last=False,
                                    pin_memory=False)

        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)



    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    # define model
    model = define_Model(opt)
    model.init_train()
    # define LPIPS function
    loss_fn_alex = lpips.LPIPS(net='alex').to(model.device)
    # define early stopping
    if opt['train']['is_early_stopping']:
        early_stopping = utils_early_stopping.EarlyStopping(patience=opt['train']['early_stopping_num'])

    # record
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    # if it is not set, keep running
    max_iter_epoch = opt['train']['max_iter_epoch'] if opt['train']['max_iter_epoch'] else 10000000000
    print(f"max iterative step: {max_iter_epoch}")

    for epoch in range(max_iter_epoch):

        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

                # record train loss
                log_wandb = {'TRAIN/step': current_step, 'TRAIN/Learning Rate': model.current_learning_rate(),}
                logger_tensorboard.add_scalar('Learning Rate', model.current_learning_rate(), global_step=current_step)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss', logs['G_loss'], global_step=current_step)
                log_wandb['TRAIN LOSS/G_loss'] = logs['G_loss']
                if 'G_loss_image' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_image', logs['G_loss_image'], global_step=current_step)
                    log_wandb['TRAIN LOSS/G_loss_image'] = logs['G_loss_image']
                if 'G_loss_frequency' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_frequency', logs['G_loss_frequency'], global_step=current_step)
                    log_wandb['TRAIN LOSS/G_loss_frequency'] = logs['G_loss_frequency']
                if 'G_loss_preceptual' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_preceptual', logs['G_loss_preceptual'], global_step=current_step)
                    log_wandb['TRAIN LOSS/G_loss_preceptual'] = logs['G_loss_preceptual']
                wandb.log(log_wandb)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            # try testing at the training beginning to pass the OOM sweep
            A = (current_step == 10) if opt['wandb']['is_sweep'] else False

            if ((current_step % opt['train']['checkpoint_test'] == 0) or A) and opt['rank'] == 0:

                # 0.001s
                timestamp = int(time.time() * 1000)

                # create folder for FID
                img_dir_tmp = os.path.join(opt['path']['images'], f'{timestamp}')
                util.mkdir(img_dir_tmp)
                img_dir_tmp_H = os.path.join(img_dir_tmp, 'tempH')
                util.mkdir(img_dir_tmp_H)
                img_dir_tmp_E = os.path.join(img_dir_tmp, 'tempE')
                util.mkdir(img_dir_tmp_E)
                img_dir_tmp_L = os.path.join(img_dir_tmp, 'tempL')
                util.mkdir(img_dir_tmp_L)

                # create result dict
                test_results = OrderedDict()
                test_results['psnr'] = []
                test_results['ssim'] = []
                test_results['lpips'] = []

                test_results['G_loss'] = []
                test_results['G_loss_image'] = []
                test_results['G_loss_frequency'] = []
                test_results['G_loss_preceptual'] = []

                for idx, test_data in enumerate(val_loader):
                    with torch.no_grad():

                        img_info = test_data['img_info'][0]
                        img_dir = os.path.join(opt['path']['images'], img_info)

                        # testing and adjust resolution
                        model.feed_data(test_data)

                        # model.check_windowsize()
                        model.test()
                        # model.recover_windowsize()

                        model.record_loss_for_val()

                        logs = model.current_log()
                        test_results['G_loss'].append(logs['G_loss'])
                        test_results['G_loss_image'].append(logs['G_loss_image'])
                        test_results['G_loss_frequency'].append(logs['G_loss_frequency'])
                        test_results['G_loss_preceptual'].append(logs['G_loss_preceptual'])

                        # acquire test result
                        results = model.current_results_gpu()

                        # calculate LPIPS (GPU | torch.tensor)
                        L_img = results['L']
                        E_img = results['E']
                        H_img = results['H']

                        current_lpips = util.calculate_lpips_single(loss_fn_alex, H_img, E_img).data.squeeze().float().cpu().numpy()

                        # calculate PSNR SSIM (CPU | np.float)
                        L_img = util.tensor2float(L_img)
                        E_img = util.tensor2float(E_img)
                        H_img = util.tensor2float(H_img)
                        current_psnr = util.calculate_psnr_single(H_img, E_img, border=0, data_range=H_img.max())
                        current_ssim = util.calculate_ssim_single(H_img, E_img, border=0, data_range=H_img.max())

                        # record metrics
                        test_results['psnr'].append(current_psnr)
                        test_results['ssim'].append(current_ssim)
                        test_results['lpips'].append(current_lpips)

                        # save samples
                        if idx < 5:
                            util.mkdir(img_dir)
                            cv2.imwrite(os.path.join(img_dir, 'ZF_{:05d}.png'.format(current_step)), np.clip(L_img, 0, 1) * 255)
                            cv2.imwrite(os.path.join(img_dir, 'Recon_{:05d}.png'.format(current_step)), np.clip(E_img, 0, 1) * 255)
                            cv2.imwrite(os.path.join(img_dir, 'GT_{:05d}.png'.format(current_step)), np.clip(H_img, 0, 1) * 255)

                        if opt['datasets']['val']['resize_for_fid']:
                            resize_for_fid = opt['datasets']['val']['resize_for_fid']
                            cv2.imwrite(os.path.join(img_dir_tmp_L, 'ZF_{:05d}.png'.format(idx)), resize(np.clip(L_img, 0, 1), (resize_for_fid[0], resize_for_fid[1])) * 255)
                            cv2.imwrite(os.path.join(img_dir_tmp_E, 'Recon_{:05d}.png'.format(idx)), resize(np.clip(E_img, 0, 1), (resize_for_fid[0], resize_for_fid[1])) * 255)
                            cv2.imwrite(os.path.join(img_dir_tmp_H, 'GT_{:05d}.png'.format(idx)), resize(np.clip(H_img, 0, 1), (resize_for_fid[0], resize_for_fid[1])) * 255)
                        else:
                            cv2.imwrite(os.path.join(img_dir_tmp_L, 'ZF_{:05d}.png'.format(idx)), np.clip(L_img, 0, 1) * 255)
                            cv2.imwrite(os.path.join(img_dir_tmp_E, 'Recon_{:05d}.png'.format(idx)), np.clip(E_img, 0, 1) * 255)
                            cv2.imwrite(os.path.join(img_dir_tmp_H, 'GT_{:05d}.png'.format(idx)), np.clip(H_img, 0, 1) * 255)

                # summarize psnr/ssim/lpips
                ave_psnr = np.mean(test_results['psnr'])
                # std_psnr = np.std(test_results['psnr'], ddof=1)
                ave_ssim = np.mean(test_results['ssim'])
                # std_ssim = np.std(test_results['ssim'], ddof=1)
                ave_lpips = np.mean(test_results['lpips'])
                # std_lpips = np.std(test_results['lpips'], ddof=1)

                # calculate FID
                if opt['dist']:
                    # DistributedDataParallel (If multiple GPUs are used to train, use the 2nd GPU for FID calculation.)
                    log = os.popen("{} -m pytorch_fid {} {} ".format(
                        sys.executable,
                        img_dir_tmp_H,
                        img_dir_tmp_E)).read()
                else:
                    # DataParallel (If multiple GPUs are used to train, use the 2nd GPU for FID calculation for unbalance of GPU menory use.)
                    if len(opt['gpu_ids']) > 1:
                        log = os.popen("{} -m pytorch_fid --device cuda:1 {} {} ".format(
                            sys.executable,
                            img_dir_tmp_H,
                            img_dir_tmp_E)).read()
                    else:
                        log = os.popen("{} -m pytorch_fid {} {} ".format(
                            sys.executable,
                            img_dir_tmp_H,
                            img_dir_tmp_E)).read()

                print(log)
                # fid = eval(log.replace('FID:  ', ''))

                # delete fid temp
                shutil.rmtree(img_dir_tmp)

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}; Average Average SSIM : {:<.4f}; LPIPS : {:<.4f};'
                            .format(epoch, current_step, ave_psnr, ave_ssim, ave_lpips))

                logger_tensorboard.add_scalar('VALIDATION PSNR', ave_psnr, global_step=current_step)
                logger_tensorboard.add_scalar('VALIDATION SSIM', ave_ssim, global_step=current_step)
                logger_tensorboard.add_scalar('VALIDATION LPIPS', ave_lpips, global_step=current_step)


                log_wandb = {'VAL/step': current_step, }
                log_wandb['VAL METRICS/PSNR'] = ave_psnr
                log_wandb['VAL METRICS/SSIM'] = ave_ssim
                log_wandb['VAL METRICS/LPIPS'] = ave_lpips


                ave_G_loss = np.mean(test_results['G_loss'])
                ave_G_loss_image = np.mean(test_results['G_loss_image'])
                ave_G_loss_frequency = np.mean(test_results['G_loss_frequency'])
                ave_G_loss_preceptual = np.mean(test_results['G_loss_preceptual'])

                logger_tensorboard.add_scalar('VALIDATION G_loss', ave_G_loss, global_step=current_step)
                logger_tensorboard.add_scalar('VALIDATION G_loss_image', ave_G_loss_image, global_step=current_step)
                logger_tensorboard.add_scalar('VALIDATION G_loss_frequency', ave_G_loss_frequency, global_step=current_step)
                logger_tensorboard.add_scalar('VALIDATION G_loss_preceptual', ave_G_loss_preceptual, global_step=current_step)

                log_wandb['VAL LOSS/G_loss'] = ave_G_loss
                log_wandb['VAL LOSS/G_loss_image'] = ave_G_loss_image
                log_wandb['VAL LOSS/G_loss_frequency'] = ave_G_loss_frequency
                log_wandb['VAL LOSS/G_loss_preceptual'] = ave_G_loss_preceptual
                wandb.log(log_wandb)

                logger.info(
                    f"[Val] <epoch: {epoch}, iter: {current_step}>, "
                    f"G_loss: {ave_G_loss:.4e}, "
                    f"G_loss_image: {ave_G_loss_image:.4e}, "
                    f"G_loss_frequency: {ave_G_loss_frequency:.4e}, "
                    f"G_loss_preceptual: {ave_G_loss_preceptual:.4e}"
                )

                # # early stopping
                # if opt['train']['is_early_stopping']:
                #     early_stopping(ave_psnr, model, epoch, current_step)
                #     if early_stopping.is_save:
                #         logger.info('Saving the model by early stopping')
                #         model.save(f'best_{current_step}')
                #     if early_stopping.early_stop:
                #         print("Early stopping!")
                #         break

    print("Training Stop")


if __name__ == '__main__':

    main()

    ############################## EXAMPLE ##############################

    ############################## PENDING ##############################

    ############################## TRAINING #############################

    ############################## TRAINED ##############################
