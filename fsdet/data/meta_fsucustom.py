import contextlib
import io
import os
import json
import numpy as np

from pycocotools.coco import COCO
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from .builtin_meta import _get_fsucustom_fewshot_instances_meta

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


__all__ = ["register_meta_fsucustom"]


def load_fsucustom_json(json_filename, root, dataset_name):
    with open(json_filename, 'r') as f:
                    json_file = json.load(f)

    data = []
    for file in json_file['annotations']:
        
        filename = json_file['images'][file['image_id']]['file_name']
        record = {
            "file_name" : f"{os.path.join(root, filename)}", # full path to image
            "image_id" :  file['image_id'], # image unique ID
            "height" : json_file['images'][file['image_id']]['height'], # height of image
            "width" : json_file['images'][file['image_id']]['width'], # width of image
            "annotations": {
                "category_id" : file['category_id'], # class unique ID
                "bbox" : file['bbox'], # bbox coordinates
                "bbox_mode" : BoxMode.XYWH_ABS, # bbox mode, depending on your format
                "iscrowd": file['iscrowd']
            }
        }
        data.append(record)
    print(data[0])
    return data

def register_meta_fsucustom(name, metadata, rootdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_fsucustom_json(annofile, rootdir, name),
    )

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=rootdir,
        evaluator_type="coco",
        dirname="datasets",
        **metadata,
    )

