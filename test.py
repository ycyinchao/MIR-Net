import argparse
import os
import sys

#sys.path.insert(0, '../')
from data.dataset import CamObjDataset
from eval.test_metrics import eval_results

sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
from torch.utils.data import DataLoader
import time
import logging as logger
from net import Net
def test(opt,Network,Datasets):
    with torch.no_grad():

        for _data_name in Datasets:
            # for _data_name in ['CAMO', 'COD10K', 'CHAMELEON','NC4K']:
            data_path = '../Dataset/TestDataset/{}/'.format(_data_name)

            final_save_path = './res/{}/final/{}/'.format(opt.pth_path.split('/')[-2], _data_name)

            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            model = Network(opt)
            model.load_state_dict(torch.load(opt.pth_path))
            model.cuda()
            model.eval()

            os.makedirs(final_save_path, exist_ok=True)

            opt.val_root = data_path
            dataset = CamObjDataset(opt)
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

            cost_time = 0
            images005 = ['COD10K-CAM-1-Aquatic-15-SeaHorse-1110', 'COD10K-CAM-1-Aquatic-7-Flounder-262', 'COD10K-CAM-2-Terrestrial-23-Cat-1328', 'COD10K-CAM-3-Flying-64-Moth-4448', 'COD10K-CAM-1-Aquatic-3-Crab-35', 'COD10K-CAM-1-Aquatic-14-ScorpionFish-896', 'COD10K-CAM-3-Flying-65-Owl-4638', 'COD10K-CAM-3-Flying-56-Cicada-3460', 'COD10K-CAM-3-Flying-50-Bat-2975', 'COD10K-CAM-2-Terrestrial-26-Chameleon-1692', 'COD10K-CAM-3-Flying-64-Moth-4452']




            for image, mask, (H, W), name in test_loader:
                if name[0].split('.')[0] not in images005:
                    continue
                start_time = time.perf_counter()
                res, out_dst = model(image.cuda().float(),(H,W),"000_100high500low_"+name[0].split('.')[0])
                cost_time += time.perf_counter() - start_time

                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # print('> {} - {}'.format(_data_name, name))
                res = 255 * res
                res = res.astype(np.uint8)

                cv2.imwrite(final_save_path + name[0], res)

            fps = len(test_loader.dataset) / cost_time
            msg = '%s len(imgs)=%s, fps=%.4f' % (_data_name, len(test_loader.dataset), fps)
            print(msg)
            logger.info(msg)

if __name__=='__main__':

    method = 'MIR-Net'
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainsize', type=int, default=384, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./checkpoints/{}/model-best.pth'.format(method))
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--val_root', type=str, default='../Dataset/TestDataset/')
    opt = parser.parse_args()

    DATASETS = ['CHAMELEON','CAMO','COD10K','NC4K']
    test(opt,Net,DATASETS)

    eval_results('./res/{}/final/'.format(method),DATASETS)