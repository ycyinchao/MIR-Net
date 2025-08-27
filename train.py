
import logging
from functools import partial
import datetime
import time

from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import logging as logger

from data.dataset import CamObjDataset
from train_processes import *
from tools import *
from net import Net

""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum


def get_polylr(base_lr, last_epoch, num_steps, power):
    return base_lr * (1.0 - min(last_epoch, num_steps-1) / num_steps) **power


def validate(model, val_loader):
    model.train(False)
    avg_mae = 0.0
    cnt = 0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            pred_fg,pred_bg = model(image)
            out_fg = F.interpolate(pred_fg[0], size=shape, mode='bilinear', align_corners=False)
            out_bg = F.interpolate(pred_bg[0], size=shape, mode='bilinear', align_corners=False)
            pred = torch.sigmoid(out_fg[0, 0]-out_bg[0, 0])
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            avg_mae += torch.abs(pred - mask[0]).mean().item()
            cnt += len(image)

    model.train(True)
    return (avg_mae / cnt)

def validate_multiloader(model, val_loader):
    maes = []
    for v in val_loader:
        st = time.time()
        mae = validate(model, v)
        maes.append(mae)
        print('Spent %.3fs, %s MAE: %s'%(time.time()-st, v.dataset.data_name, mae))
        logging.info('Spent %.3fs, %s MAE: %s'%(time.time()-st, v.dataset.data_name, mae))
    return sum(maes)/len(maes)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=50):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def train(train_loader,val_loaders,model,optimizer,writer,opt, train_loss):
    global global_step
    global et
    global min_mae

    for epoch in range(1, opt.epoch+1):
        for batch_idx, data in enumerate(train_loader):
            image,mask,_,__ = data
            image = image.float().cuda()
            mask = mask.cuda()

            st = time.time()
            niter = epoch * db_size + batch_idx-1   # 减一是为了最后一个epoch的lr不会报错：ZeroDivisionError: float division by zero

            cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

            global_step += 1

            loss2, loss3, loss4, loss5 = train_loss(image, mask, model)

            ######  objective function  ######
            loss = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            writer.add_scalar('loss', loss.item(), global_step=global_step)

            ta = time.time() - st
            et = 0.9*et + 0.1*ta if et>0 else ta
            if batch_idx % 10 == 0:
                msg = '%s | eta:%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f' % (datetime.datetime.now(), datetime.timedelta(seconds = int((opt.epoch*db_size-niter)*et)), global_step, epoch, opt.epoch, cur_lr, loss.item(), loss2.item(), loss3.item(), loss4.item())
                print(msg)
                logger.info(msg)

        # if epoch > opt.epoch // 2  or epoch % 10 == 0:
        if epoch>0:
            mae = validate_multiloader(model, val_loaders)
            print('VAL MAE:%s' % (mae))
            logging.info('VAL MAE:%s' % (mae))
            writer.add_scalar('val', mae, global_step=global_step)
            if mae < min_mae :
                min_mae = mae
                print('best epoch is:%d, MAE:%s' % (epoch, min_mae))
                logging.info('best epoch is:%d, MAE:%s' % (epoch, min_mae))
                torch.save(model.state_dict(), opt.save_path + '/model-best.pth')
            
        # if epoch == opt.epoch or (epoch) % 10 == 0:
        #     torch.save(model.state_dict(), opt.save_path + '/model-' + str(epoch))

if __name__=='__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='../Dataset/TrainDataset',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='../Dataset/TestDataset/CAMO',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./checkpoints/MIR-Net/',
                        help='the path to save model and log')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    cudnn.benchmark = True
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'# 用于调试，强制 CUDA 操作同步执行以便更好地发现错误。会导致代码执行速度非常慢

    ## dataset
    dataset = CamObjDataset(opt)
    train_loader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=8)

    opt.mode = 'test'
    dataset = CamObjDataset(opt)
    val_loaders = [DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)]
    opt.mode = 'train'

    min_mae = 1.0

    ## network
    model = Net(opt).cuda()
    # model = Net(opt)
    # model = nn.DataParallel(model).cuda()
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")

    params = model.parameters()
    print('model paramters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info("model paramters:" + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))


    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    ## log
    writer = SummaryWriter(save_path + 'summary')

    db_size = len(train_loader)
    global_step = 0
    et = 0

    from net import Net
    tm = partial(train_loss)
    train(train_loader,val_loaders,model,optimizer,writer,opt, tm)