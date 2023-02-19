# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """
    bx = (2*sigmoid(tx)-0.5)+cx (-0.5, 1.5)
    by = (2*sigmoid(ty)-0.5)+cy (-0.5, 1.5)
    bw = pw*(2*sigmoid(tw))^2 (0, 4)
    bh = ph*(2*sigmoid(th))^2 (0, 4)
    """
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # BCEWithLogitsLoss combine Sigmoid & BCE
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # opt.label_smoothing default value is 0.0
        # default values of self.cp and self.cn are 1.0 and 0.0
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        # dictionary.get(keyname, value) if the key doesn't exit, it will return the value
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7 balance different head loss
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """
        :param p:
        :param targets:
        :return:
        """
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        # indices [image batch id, anchor id, gj, gi]
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        # during training, p is a list when contain each output head
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor_index, gridy, gridx
            # pi.shape: [b, 3, imgsz/stride, imgsz/stride, 5+nc]
            # tobj initialized with 0 represents non-object
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of objects
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression for bounding box with CIoU loss
                pxy = pxy.sigmoid() * 2 - 0.5  # relative to each grid cell (-0.5, 1.5)
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]  # absolute width and height value
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # pbox.shape = tbox[i].shape = [num_gt, 4]
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                # sort_obj_iou default value is False
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                # gr default value is 1.0
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                # tobj initialized with torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
                tobj[b, a, gj, gi] = iou  # iou ratio for objectness loss

                # Classification
                # Because object loss can be updated for one class
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # self.cn default value is 0.0
                    # def full_like(input: Tensor, fill_value: int | float | bool)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    # the default value of self.cp is 1.0
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """
        :param p: list
        :param targets: list targets(image,class,x,y,w,h) with shape [nt, 6] x, y, w, h is ratio value
        :return: indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
                 tbox.append(torch.cat((gxy - gij, gwh), 1))  gxy-gij is tx and ty gwh is absolute value
                 anch.append(anchors[a])  # anchors
                 tcls.append(c)  # class
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h) with shape [nt, 6]
        # na = 3
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # 7 represents (image, class, x, y, w, h, anchor_index)
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        # ai.shape: (na, nt)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets.shape: [na, nt, (image, class, x, y, w, h ,anchor)]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        # (-0.5, 1.5)
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            # anchors and p are result lists from each head
            anchors, shape = self.anchors[i], p[i].shape  # p[i].shape: [bs, 3, imgsz/stride, imgsz/stride, 5+nc]
            # [3, 2, 3, 2] is the index for [imgsz/stride, imgsz/stride, imgsz/stride, imgsz/stride]
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain for affining to feature map

            # Match targets to anchors
            # affine ground truth to feature map
            t = targets * gain  # shape(3,n,7), then the origin of coordinates t is the left upper
            if nt:
                # Matches by gtw/aw and gth/ah smaller than anchor_t
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # check the r and 1/r whether smaller than hpy['anchor_t']
                # compare: tensor.max(dim=None, keepdim=False) return values, indices
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter anchor,

                # Offsets https://blog.csdn.net/qq_37541097/article/details/123594351
                gxy = t[:, 2:4]  # grid xy shape=[after_anchor_filter, 2]
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # to check the center point at the upper part or lower part
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # [5, after_anchor_filter]
                # input t.shape = [after_anchor_filter, 7] output t.shape
                t = t.repeat((5, 1, 1))[j]  # 5 represents center, up, down, right, left, after selecting ground truth cell
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # When use None as index, it will unsqueeze() or expand_dims()
            else:
                t = targets[0]
                offsets = 0

            # Define
            # tensor.chunk(chunks, dim=1)
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            # b: image batch id, a: anchor id, gj: ground truth cell j incex, gi: ground truth cell i index
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # gxy is relative to each grid cell, belong [0, 1]
            # gwh is the absolute length from t = targets * gain
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
