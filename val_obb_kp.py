# Based on the YOLOv5 HBB validation, I change it for validating OBB
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from threading import Thread

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
# from utils.dataloaders import create_dataloader
from utils.dataloaders_obb_kp import create_dataloader
from utils.general_obb_kp import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                                  check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                                  print_args, scale_boxes, xywh2xyxy, xyxy2xywh, non_max_suppression_obb, scale_polys)
# from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.metrics_obb_kp import ConfusionMatrix, ap_per_class, box_iou
# from utils.plots import output_to_target, plot_images, plot_val_study
from utils.plots_obb_kp import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
from utils.obb_utils import rbox2poly, poly2hbb


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result with (cls, *xywh, conf) or (cls, *xywh)
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(pred_hbbn, pred_polyn, jdict, path, class_map):
    """
    Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236, "poly": [...]}
    Args:
        pred_hbbn (tensor): (n, [poly, conf, cls])
        pred_polyn (tensor): (n, [xyxy, conf, cls])
    """
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(pred_hbbn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(pred_polyn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[-1])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[-2], 5),
            'poly': [round(x, 1) for x in p[:8]],
            'file_name': path.stem
        })


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)  # [num_detection, 10]
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=1500,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    # During training process of train_obb.py, it will use model=ema.ema,
    # while it uses attempt_load() for model
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # torch.linspace(start, end, step)
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()  # number different iou threshold for calculating mAP

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # data[task] get different path for training or validating
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]  # return dataloader, dataset

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)  # initialization
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    # tp: true positive; fp: false positive
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    # loss = torch.zeros(3, device=device)
    loss = torch.zeros(5, device=device)  # (box, obj, cls, angle, pos)
    # for stats, it will save (correct, conf, pcls, tcls)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # img(tensor): (3, height, width), RGB
        # labels_out(tensor): (n, [None clsid cx cy l s gaussian_θ_labels])
        # θ∈[-pi / 2, pi / 2)
        # img_file(str): img_dir
        # shapes: None or [(h_raw, w_raw), ((h_ratios, w_ratios), wh_paddings)], for COCO mAP rescaling
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            # compute_loss is passed by compute_loss
            # x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
            # preds: torch.cat(z, 1), train_out: x
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            # return (lbox + lobj + lcls + langle + lpos) * bs, torch.cat((lbox, lobj, lcls, langle, lpos)).detach()
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls, angle, pos

        # NMS
        # as for my finger nail dataset,
        # the ground truth (x1, y1, x2, y2, x3, y3, x4, y4) values are not normalized by image size
        # targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # lb:-> list
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            # input-> prediction (tensor): (b, n_all_anchors, [cx cy l s obj num_cls angle_cls])
            # output-> list of detections,
            # len=batch_size, on (n,7) tensor per image [xylsθ, conf, cls] θ ∈ [0, 180)
            preds = non_max_suppression_obb(preds,
                                            conf_thres,
                                            iou_thres,
                                            labels=lb,
                                            multi_label=True,  # each bounding boxes can represent several classes
                                            agnostic=single_cls,
                                            max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            # targets:  [index_of_batch clsid cx cy l s gaussian_θ_labels]
            labels = targets[targets[:, 0] == si, 1:]  # labels (n_gt, [clsid, cx, cy, l, s, gaussian_θ_labels])
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init with correct.shape [npr, niou]
            seen += 1

            if npr == 0:
                if nl:
                    # (correct.shape: [npr, niou], [2,0], labels[:, 0].shape: [nl,1]
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))  # labels[:, 0] is cls_id
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                # pred[:, 5] = 0
                pred[:, 6] = 0  # [x,y,l,s,angle,conf,cls]
            poly = rbox2poly(pred[:, :5])  # poly (n, 8)
            pred_poly = torch.cat((poly, pred[:, -2:]), dim=1)  # (n, [poly, conf, cls])
            # "poly2hbb" will get the minimum outer rectangle
            hbbox = xywh2xyxy(poly2hbb(pred_poly[:, :8]))  # (n, [x1 y1 x2 y2])
            pred_hbb = torch.cat((hbbox, pred_poly[:, -2:]), dim=1)  # (n, [xyxy, conf, cls])

            pred_polyn = pred_poly.clone()  # predn (tensor): (n, [poly, conf, cls]
            # shapes: None or [(h_raw, w_raw), ((h_ratios, w_ratios), wh_paddings)], for COCO mAP rescaling
            # shape: shapes[si][0]
            # scale the predicted bounding boxes to original image
            scale_polys(im[si].shape[1:], pred_polyn[:, :8], shape, shapes[si][1])  # native space
            hbboxn = xywh2xyxy(poly2hbb(pred_polyn[:, :8]))
            pred_hbbn = torch.cat((hbboxn, pred_polyn[:, -2:]), dim=1)

            # Evaluate
            if nl:
                # labels [clsid, x, y, l, s, gaussian_a_cls]
                # keepdim=True can promise the max_index has the same tensor shape as the input
                _, angle_labels = torch.max(labels[:, 5:], 1, keepdim=True)
                rlabels = torch.cat((labels[:, 1:5], angle_labels), dim=1)
                tpoly = rbox2poly(rlabels)
                scale_polys(im[si].shape[1:], tpoly, shape, shapes[si][1])  # native space labels
                tbox = xywh2xyxy(poly2hbb(tpoly))
                labels_hbbn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels with [cls, x1, y1, x2, y2]
                correct = process_batch(pred_hbbn, labels_hbbn, iouv)
                if plots:
                    confusion_matrix.process_batch(pred_hbbn, labels_hbbn)
            stats.append(
                (correct, pred_poly[:, 8], pred_poly[:, 9], labels[:, 0]))  # (correct, conf, predicted cls, target cls)

            # Save/log
            if save_txt:
                save_one_txt(pred_hbbn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(pred_hbbn, pred_polyn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred_hbb, pred_hbbn, path, names, im[si])

        # Plot images for validation valid_batch(id)_labels & prediction,
        # for the train_batch(id), it will on the Loggers class
        if plots and batch_i < 3:
            # the im is not the original image of the dataset
            Thread(plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names),
                   daemon=True).start()  # labels
            Thread(plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names),
                   daemon=True).start()  # pred
            # plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            # plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    # stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, predicted cls, target cls)
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path('../datasets/coco/annotations/instances_val2017.json'))  # annotations
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            subprocess.run('zip -r study.zip study_*.txt'.split())
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
