import os
import json
import numpy as np
import cv2
import time
import datetime
from datetime import datetime
from PIL import Image
from shutil import move

def move_images(source_rootdir, dest_rootdir):
    for root, dirs, files in os.walk(source_rootdir):
        destdir = ''
        for afile in files:
            if 'train' in root or 'val' in root:
                if 'train' in root:
                    destdir = 'train'

                elif 'val' in root:
                    destdir = 'val'
                else:
                    print(root)
                source = os.path.join(root, afile)
                dest = os.path.join(dest_rootdir, f"{destdir}", afile)
                move(source, dest)

#move_images('/home/cspooner/LAS/few-shot-object-detection/datasets/custom/Cylinder', '/home/cspooner/LAS/few-shot-object-detection/datasets/custom/')


def split_annotations(annotation_file, trainlist, vallist, JSON_FILENAME):
    
    #info
    info = annotation_file['info']

    train_info = dict(info)
    val_info = dict(info)

    train_info['split_type'] = "train"
    val_info['split_type'] = 'val'

    #categories
    categories = annotation_file['categories']
    #licenses
    licenses = annotation_file['licenses']

    #images
    train_id_img = 0
    val_id_img = 0
    train_image_files = []
    val_image_files = []
    train_indices = []
    val_indices = []

    imageID_old2new = {}
    imageID_name = {}
    
    for im in annotation_file['images']:
        #print(im['id'])
        if im['file_name'] in trainlist:
            #print(f"{im['file_name']} is in the trainlist")

            train_indices.append(im['id'])

            imageID_old2new[im['id']] = train_id_img
            imageID_name[train_id_img] = im['file_name']
            im['id'] = train_id_img
            train_image_files.append(im)
            train_id_img += 1
        elif im['file_name'] in vallist:
            #print(f"{im['file_name']} is in the vallist")
            val_indices.append(im['id'])
            imageID_old2new[im['id']] = val_id_img
            imageID_name[val_id_img] = im['file_name']
            im['id'] = val_id_img
            val_image_files.append(im)
            val_id_img += 1
        else:
            print(f"{im['file_name']} is not in either")
           
    #annotations
    train_ann_files = []
    val_ann_files = []
    splitted = ''

    for ann in annotation_file['annotations']:
       
        
        oldAnn = ann['image_id']
        ann['image_id'] = imageID_old2new[oldAnn]
        file_name = imageID_name[imageID_old2new[oldAnn]]

        if oldAnn in train_indices:
        #if file_name in trainlist:
            train_ann_files.append(ann)
            splitted = 'train'

        elif oldAnn in val_indices:
        #elif file_name in vallist:
            val_ann_files.append(ann)
            splitted = 'val'


    final_trainDict = {"info": train_info, "licenses":licenses, "images":train_image_files, "annotations":train_ann_files, 'categories':categories}

    # Serializing json
    train_json_object = json.dumps(final_trainDict, indent=2)
    trainjsonFileName = f'train_{JSON_FILENAME}'

    # Writing to sample.json
    with open(trainjsonFileName, "w") as outfile:
        outfile.write(train_json_object)
            
    final_valDict = {"info": val_info, "licenses":licenses, "images":val_image_files, "annotations":val_ann_files, 'categories':categories}
  
    # Serializing json
    val_json_object = json.dumps(final_valDict, indent=2)
    valjsonFileName = f'val_{JSON_FILENAME}'

    # Writing to sample.json
    with open(valjsonFileName, "w") as outfile:
        outfile.write(val_json_object)

def driver_code():


    ROOTWALKDIR = r"/home/cspooner/LAS/few-shot-object-detection/datasets/custom/"
    JSON_FILENAME = 'FSU_ISL_synthetic_LAS_CCTV.json'
    #JSON_FILENAME = '/home/cspooner/LAS/few-shot-object-detection/datasets/lvis/lvis_v1_val.json'
    with open(JSON_FILENAME, 'r') as f:
        data = json.load(f)

    train = os.listdir(os.path.join(ROOTWALKDIR, 'train'))
    val = os.listdir(os.path.join(ROOTWALKDIR, 'val'))
    split_annotations(data, train, val, JSON_FILENAME)
        

    
        

if __name__ == "__main__":
    driver_code()

