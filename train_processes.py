from feature_loss import *
from tools import *
from utils import ramps

criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()

loss_lsc = FeatureLoss().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)
    
def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        pp = Crop(0.7, 0.7)
    return pp

def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0,1), padding=(1,1), groups=c)
    return c1, b

def unsymmetric_grad(x, y, calc, w1, w2):
    '''
    x: strong feature
    y: weak feature'''
    return calc(x, y.detach())*w1 + calc(x.detach(), y)*w2

# targeted at boundary: only p/n coexsits.
# learn features that focus on boundary prediction
# use feature vectors to guide pixel prediction 
# covariance to encourage the feature difference in the most decisive ones
def feature_loss(feature_map, pred, kr=4, norm=False, crtl_loss=True, w_ftp=0, topk=16, step_ratio=2):
    '''
    pred: n, 1, h, w'''
    # normalize feature map (but how?)
    if norm:
        fmap = feature_map / feature_map.std(dim=(-1,-2), keepdim=True).mean(dim=1, keepdim=True)
    else: fmap=feature_map
    # print(fmap.max(), fmap.min(), fmap.std(dim=(-1,-2)).max())

    n, c, h, w =fmap.shape
    # get local feature map
    ks = 2*kr
    assert h%ks==0 and w%ks==0
    # print('ks', ks)
    uf = lambda x: F.unfold(x, ks, padding = 0, stride=ks//step_ratio).permute(0,2,1).reshape(-1, x.shape[1], ks*ks) # N * no.blk, 64, 8*8
    fcmap = uf(fmap) 
    fcpred = uf(pred) # N', 1, 10*10
    # # get fg/bg confident coexisting block
    cfd_thres = .8
    exst = lambda x: (x>cfd_thres).sum(2, keepdim=True) > 0.3*ks*ks
    coexists = (exst(fcpred) & exst(1-fcpred))
    coexists = coexists[:, 0, 0] # N', 1, 1
    fcmap = fcmap[coexists]
    fcpred = fcpred[coexists]
    # print(fcmap.shape, fcpred.shape)
    if not len(fcmap):
        return 0, 0
    # minus mean
    mfcmap = fcmap - fcmap.mean(2, keepdim=True)
    mfcpred = fcpred - fcpred.mean(2, keepdim=True)
    # get most relevance in confident area bout saliency
    cov = mfcmap.matmul(mfcpred.permute(0, 2, 1)) # N', 64, 1
    sgnf_id = cov.abs().topk(topk, dim=1)[1].expand(-1,-1,ks*ks) # n', topk, 10*10
    sg_fcmap = fcmap.gather(dim=1, index=sgnf_id) # n', topk, 10*10
    # different potential calculation
    crf_k = lambda x: (-(x[:, :, None]-x[:, :, :, None])**2 * 0.5).sum(1, keepdim=True).exp() # n', 1, 100, 100
    pred_grvt = lambda x,y: (1-x)*y + x*(1-y) # (x-y).abs() # x*y + (1-x)*(1-y) - x*(1-y) - (1-x)*y
    ft_grvt = lambda x: 1-crf_k(x)
    # position
    xy = torch.stack(torch.meshgrid(torch.arange(ks, device=pred.device), torch.arange(ks, device=pred.device))) / 6
    xy = (xy).reshape(1,2, ks*ks).expand(len(sg_fcmap),-1,-1) # 1, 1, 100
    ffxy = crf_k(xy)
    if crtl_loss:
        # train the feature map without pred grad
        # L2 norm loss
        pmap = fcpred.detach()
        pmap = 0.5 - pred_grvt(pmap.unsqueeze(2), pmap.unsqueeze(-1)) # n', 1, 100, 100
        fpmap = ft_grvt(sg_fcmap) * ffxy
        ice = (pmap*fpmap).mean()
        # reversely, train the pred map
        # calculate CRF with confident point
        fffm = crf_k(sg_fcmap.detach())
        kernel = fffm*ffxy # n', 1, 10*10, 10*10
    else:
        ice = 0
        fffm = crf_k(sg_fcmap)
        kernel = fffm*ffxy # n', 1, 10*10, 10*10
        kernel[torch.eye(ks*ks, device=pred.device, dtype=bool).expand_as(kernel)] = 0

    pp = pred_grvt(fcpred[:,:,None], fcpred.unsqueeze(-1)) # n', 1, 100, 100
    if w_ftp==0:
        crf = (kernel * pp).mean()
    elif w_ftp==1:
        crf = (kernel.detach() * pp).mean() * (1+w_ftp)
    else:
        crf = unsymmetric_grad(kernel, pp, lambda x,y:(x*y).mean(), 1-w_ftp, 1+w_ftp)
    return crf, ice

def train_loss(image, mask, net, gamma_hyper_param__loss=0.3, l=0.3):


    pre_transform = get_transform(ops=[0,1,2])
    image_tr = pre_transform(image)

    scale_factor = np.random.choice([0.25, 0.5, 0.75])
    image_scale = F.interpolate(image_tr, scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def out_proc(out2, out3, out4, out5):
        a = [out2, out3, out4, out5]
        a = [i.sigmoid() for i in a]
        return a
    pred_fg,pred_bg = net(image, )
    out4_fg,out3_fg,out2_fg,out1_fg = out_proc(*pred_fg)
    out4_bg,out3_bg,out2_bg,out1_bg = out_proc(*pred_bg)

    pred_fg_s,pred_bg_s = net(image_scale, )
    out4_fg_s,out3_fg_s,out2_fg_s,out1_fg_s = out_proc(*pred_fg_s)
    out4_bg_s,out3_bg_s,out2_bg_s,out1_bg_s = out_proc(*pred_bg_s)

    out4_fg_ss = pre_transform(out4_fg)
    out4_fg_scale = F.interpolate(out4_fg_ss, scale_factor=scale_factor, mode='bilinear', align_corners=True)

    out4_bg_ss = pre_transform(out4_bg)
    out4_bg_scale = F.interpolate(out4_bg_ss, scale_factor=scale_factor, mode='bilinear', align_corners=True)
    ########################################################################################################## CV Loss
    loss_ssc = (SaliencyStructureConsistency(out4_fg_s, out4_fg_scale.detach(), 0.85) * (gamma_hyper_param__loss + 1) + SaliencyStructureConsistency(out4_fg_s.detach(), out4_fg_scale, 0.85) * (1 - gamma_hyper_param__loss))
    loss_ssc += (SaliencyStructureConsistency(out4_bg_s, out4_bg_scale.detach(), 0.85) * (gamma_hyper_param__loss + 1) + SaliencyStructureConsistency(out4_bg_s.detach(), out4_bg_scale, 0.85) * (1 - gamma_hyper_param__loss))

    gt = mask.squeeze(1).long()
    bg_label = gt.clone()
    fg_label = gt.clone()
    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255


    ######   local saliency coherence loss (scale to realize large batchsize)  ######
    image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
    sample = {'rgb': image_}
    # print('sample :', image_.max(), image_.min(), image_.std())
    out4_fg_ = F.interpolate(out4_fg, scale_factor=0.25, mode='bilinear', align_corners=True)
    loss2_lsc = loss_lsc(out4_fg_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    out4_bg_  = F.interpolate(out4_bg, scale_factor=0.25, mode='bilinear', align_corners=True)
    loss2_lsc += loss_lsc(out4_bg_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    out4 = torch.cat((out4_bg,out4_fg), 1)
    loss2 = loss_ssc + criterion(out4, fg_label) + criterion(out4, bg_label) + l * loss2_lsc ## dominant loss

    ######  auxiliary losses  ######
    out3_fg_ = F.interpolate(out3_fg, scale_factor=0.25, mode='bilinear', align_corners=True)
    loss3_lsc = loss_lsc(out3_fg_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    out3_bg_ = F.interpolate(out3_bg, scale_factor=0.25, mode='bilinear', align_corners=True)
    loss3_lsc += loss_lsc(out3_bg_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    out3 = torch.cat((out3_bg,out3_fg), 1)
    loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * loss3_lsc

    out4_fg_ = F.interpolate(out2_fg, scale_factor=0.25, mode='bilinear', align_corners=True)
    loss4_lsc = loss_lsc(out4_fg_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    out4_bg_ = F.interpolate(out2_bg, scale_factor=0.25, mode='bilinear', align_corners=True)
    loss4_lsc += loss_lsc(out4_bg_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    out2 = torch.cat((out2_bg,out2_fg), 1)
    loss4 = criterion(out2, fg_label) + criterion(out2, bg_label) + l * loss4_lsc

    out5_fg_ = F.interpolate(out1_fg, scale_factor=0.25, mode='bilinear', align_corners=True)
    loss5_lsc = loss_lsc(out5_fg_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    out5_bg_ = F.interpolate(out1_bg, scale_factor=0.25, mode='bilinear', align_corners=True)
    loss5_lsc += loss_lsc(out5_bg_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    out1 =  torch.cat((out1_bg,out1_fg), 1)
    loss5 = criterion(out1, fg_label) + criterion(out1, bg_label) + l * loss5_lsc

    return loss2, loss3, loss4, loss5