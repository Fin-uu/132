import os
import time
import importlib
from pathlib import PurePath

import torch

import numpy as np

from monai.data import decollate_batch
from monai.transforms import (
    LoadImaged,
    #EnsureChannelFirstD,
    SqueezeDimd,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Compose,
    LabelFilter,
    MapLabelValue,
    Spacing,
    SqueezeDim
)
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric,get_confusion_matrix, compute_confusion_matrix_metric

from data_utils.io import save_img
import matplotlib.pyplot as plt


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
        output = torch.argmax(output, dim=1)
    return output


def check_channel(inp):
    # 此函數確保輸入張量為 5D: (Batch, Channel, Depth, Height, Width)
    len_inp_shape = len(inp.shape)
    
    if len_inp_shape == 4:
        # 如果是 4D (B, D, H, W)，在 dim=1 插入 C 維度
        inp = torch.unsqueeze(inp, dim=1) # 結果: (B, 1, D, H, W) -> 5D
        
    elif len_inp_shape == 3:
        # 如果是 3D (D, H, W)，插入 C (dim=0) 和 B (dim=0) 維度
        inp = torch.unsqueeze(inp, dim=0) # (1, D, H, W)
        inp = torch.unsqueeze(inp, dim=0) # (1, 1, D, H, W)
    
    return inp


def eval_label_pred(data, cls_num, device):
    # post transform
    post_label = AsDiscrete(to_onehot=cls_num)
    
    # metric
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )

    iou_metric = MeanIoU(include_background=False)
    
    confusion_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name="sensitivity", 
        compute_sample=False, 
        reduction="mean", 
        get_not_nans=False
    )
    
    # batch data
    val_label, val_pred = (data["label"].to(device), data["pred"].to(device))
    
    # check shape is 5
    val_label = check_channel(val_label)
    val_pred = check_channel(val_pred)
    
    # deallocate batch data
    val_labels_convert = [
        post_label(val_label_tensor) for val_label_tensor in val_label
    ]
    val_output_convert = [
        post_label(val_pred_tensor) for val_pred_tensor in val_pred
    ]
    
    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    iou_metric(y_pred=val_output_convert, y=val_labels_convert)
    confusion_metric(y_pred=val_output_convert, y=val_labels_convert)

    dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    iou_vals = iou_metric.get_buffer().detach().cpu().numpy().squeeze()
    
    confusion_vals = confusion_metric.get_buffer().detach().cpu().numpy().squeeze()
    print("Confusion_Vals：", confusion_vals)
    tp = confusion_vals[:, 0]
    fp = confusion_vals[:, 1]
    tn = confusion_vals[:, 2]
    fn = confusion_vals[:, 3]
    sensitivity_vals = tp / (tp + fn)
    specificity_vals = tn / (tn + fp)
    
    
    return dc_vals, iou_vals, sensitivity_vals, specificity_vals


def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def get_label_transform(data_name, keys=['label']):
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


def run_infering(
    model,
    data,
    model_inferer,
    post_transform,
    args):
    
    ret_dict = {}

    # 關鍵備份步驟
    original_image_meta = data.get('image_meta_dict', {}).copy() 
    original_label_meta = data.get('label_meta_dict', {}).copy()

    # test (其餘部分保持不變)
    start_time = time.time()
    data['pred'] = infer(model, data, model_inferer, args.device)
    end_time = time.time()
    ret_dict['inf_time'] = end_time-start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')

    # post process transform (保持不變)
    if args.infer_post_process:
        print('use post process infer')
        applied_labels = np.unique(data['pred'].flatten())[1:]
        data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(data['pred'])

    # eval infer tta (保持不變)
    if 'label' in data.keys():
        tta_dc_vals, tta_iou_vals, _ , _ = eval_label_pred(data, args.out_channels, args.device)
        print('infer test time aug:')
        print('dice:', tta_dc_vals)
        print('iou:', tta_iou_vals)
        ret_dict['tta_dc'] = tta_dc_vals
        ret_dict['tta_iou'] = tta_iou_vals

        # post label transform (保持不變)
        sqz_transform = SqueezeDimd(keys=['label'])
        data = sqz_transform(data)

    # post transform (保持不變)
    data = post_transform(data)

    # ----------------------------------------------------
    # eval infer origin (最複雜的區塊)
    # ----------------------------------------------------

    # 預先初始化原始指標 (必須確保 ret_dict 在任何情況下都完整)
    ori_dc_vals = ori_iou_vals = ori_sensitivity_vals = ori_specificity_vals = np.array([0])
    file_path = None
    
    if 'label' in data.keys():
        
        # 1. 嘗試確定文件路徑
        if original_label_meta and 'filename_or_obj' in original_label_meta:
            file_path = original_label_meta['filename_or_obj']
        elif original_image_meta and 'filename_or_obj' in original_image_meta:
            file_path = original_image_meta['filename_or_obj']

        # 2. 只有找到文件路徑時才繼續評估
        if file_path is not None:
            # get orginal label
            lbl_dict = {'label': file_path} # <--- 使用確定的 file_path 變數
            label_loader = get_label_transform(args.data_name, keys=['label'])
            lbl_data = label_loader(lbl_dict)
            
            data['label'] = lbl_data['label']
            data['label_meta_dict'] = lbl_data.get('label_meta_dict', original_label_meta)
            
            # 執行原始評估
            ori_dc_vals, ori_iou_vals, ori_sensitivity_vals, ori_specificity_vals = eval_label_pred(data, args.out_channels, args.device)
        else:
            print("Error: Cannot find any meta_dict for file path reconstruction. Using 0 for metrics.")

    # 3. 寫入結果 (無論是否評估成功，都要寫入 ret_dict)
    print('infer test original:')
    print('dice:', ori_dc_vals)
    print('iou:', ori_iou_vals)
    print('sensitivity:', ori_sensitivity_vals)
    print('specificity:', ori_specificity_vals)
    
    ret_dict['ori_dc'] = ori_dc_vals
    ret_dict['ori_iou'] = ori_iou_vals
    ret_dict['ori_sensitivity'] = ori_sensitivity_vals
    ret_dict['ori_specificity'] = ori_specificity_vals
    
    # ... (mmwhs 區塊保持不變)
    if args.data_name == 'mmwhs':
        mmwhs_transform = Compose([
            LabelFilter(applied_labels=[1, 2, 3, 4, 5, 6, 7]),
            MapLabelValue(orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                          target_labels=[0, 500, 600, 420, 550, 205, 820, 850]),
        ])
        data['pred'] = mmwhs_transform(data['pred'])
    
    
    if not args.test_mode:
        # save pred result
        filename = get_filename(data)
        infer_img_pth = os.path.join(args.infer_dir, filename)

        save_img(
          data['pred'], 
          original_image_meta, 
          infer_img_pth
        )
        
    return ret_dict