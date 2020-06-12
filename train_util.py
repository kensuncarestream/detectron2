import cv2
import os
from detectron2.structures import BoxMode
import itertools
import pandas as pd
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou 


def cal_iou_from_matched_file(gt_file, pred_file, threshold = 0.7):
    content = []
    boxA = []
    boxB = []
    predA = []
    predB = []
    result = {}
    
    filea = open(gt_file,"r")
    fileb = open(pred_file,"r")
    for linea in filea:
        line_split = linea.strip().split(',')
        (filename, x1, y1, x2, y2, objectness) = line_split
        if filename not in content:
            content.append(filename)
            boxA.append([int(x1), int(y1), int(x2), int(y2)])
            predA.append(int(objectness))
            
    for lineb in fileb:
        line_split = lineb.strip().split(',')
        (filename, x1, y1, x2, y2, objectness) = line_split
        if filename in content:
            boxB.append([int(x1), int(y1), int(x2), int(y2)])
            predB.append(int(objectness))
    
    sum = 0.0
    count = 0
    for index in range(len(boxA)):
        s = iou(boxA[index], boxB[index])
        if s > threshold and predA[index] == predB[index]:
            count += 1
        else: 
            print(content[index])
            print('IOU: {}'.format(s))
            print('ground_truth {}'.format(predA[index]))
            print('predicted {}'.format(predB[index]))
        # result[content[index]] = (s, boxA[index], boxB[index])
        result[content[index]] = [s] + list(itertools.chain(boxA[index], boxB[index])) + [predA[index]] + [predB[index]]
        sum = sum + s  
    mean = sum/len(boxA)
    accuracy = count/len(boxA)
    print('mean IOU: {}'.format(mean))
    print('Class accuracy: {}'.format(accuracy))
    return result


def get_annotation(datapath, annotation):
    # dict field refer to https://detectron2.readthedocs.io/tutorials/datasets.html
    # annotation is information for bounding box
    # objectness is information for classes, for example
    # 0 is knee center without medal implants
    # 1 is knee center with medal implants
    class_name = 'Knee'
    all_imgs = {}
    with open(annotation,'r') as f:
        print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            # limited to single knee, only one box per image
            # (img_id, x1, y1, x2, y2) = line_split
            (img_id, x1, y1, x2, y2, objectness) = line_split
            if img_id not in all_imgs:
                all_imgs[img_id] = {}
                all_imgs[img_id]['image_id'] = img_id
                img_filepath = os.path.join(datapath, img_id)
                assert os.path.exists(img_filepath)
                all_imgs[img_id]['file_name'] = img_filepath            
                img = cv2.imread(img_filepath)
                (rows,cols) = img.shape[:2]                
                all_imgs[img_id]['width'] = cols
                all_imgs[img_id]['height'] = rows
                all_imgs[img_id]['annotations'] = []
                
            all_imgs[img_id]['annotations'].append(
                {'bbox': [int(x1), int(y1), int(x2), int(y2)],
                 'bbox_mode': BoxMode.XYXY_ABS,
                 "category_id": int(objectness),
                }
            )
    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])
    return all_data

