import os
import json
import numpy as np
import cv2
import time
import datetime
from datetime import datetime
from PIL import Image

def create_image_mask_dictionary(segImage):
    """
    This function will create a dictionary that maps each color of an instance segmentation image
    to a binary mask.

    Parameters
    _________

    segImage : image object
        Should be an instance segmentation image


    Returns
    ________

    collections : dict
        the dictionary that maps each individual color in the image to a binary mask for that color.

    """
    #if the image was opened in opencv, which opens them in bgr, then flip everything

    if hasattr(segImage, 'shape'):
        h, w, _ = segImage.shape
        segImage = cv2.cvtColor(segImage, cv2.COLOR_BGR2RGB)

    else:
        w, h = segImage.size

    # extract all of the unique colors
    object_seg = np.array(segImage)
    colors = np.unique(object_seg.reshape(-1, object_seg.shape[2]), axis=0)

    collections = {}
    for k in colors:
        color = tuple(k)
        if color not in collections.keys():
            mask = cv2.inRange(object_seg, k, k)
            collections[color] = mask
    return collections

def get_color(colorstring):
    #convert colorstring to individual r, g, b and then flop
    # r and b for opencv
    colorstring = colorstring.replace('(', '').replace(')', '')
    r, g, b, _ = colorstring.split(',')
    r = int(r.replace('R=', ''))
    g = int(g.replace('G=', ''))
    b = int(b.replace('B=', ''))
    return((r, g, b))

def read_json(filename):
    # Opening JSON file
    with open(filename) as f:
        colorMap = {}
        # returns JSON object as
        # a dictionary
        data = json.load(f)

        # Iterating through the json
        # list
        for i in data:

            c = get_color(i['Color'])
            #print(f'Color for {i["Mesh"]} is {c}')
            if c not in colorMap:
                colorMap[c] = i['Mesh']
            else:
                print('I have seen that color before')
                if (colorMap[c] != i["Mesh"]):
                    print(f'{colorMap[c]} and {i["Mesh"]} do not match for {filename}, although they both have {c} assigned to them')
                    return
    return colorMap

def datasetInfo(version, description, contributor, date=None):
    """
    This function will create a dictionary that collects all of the information 
    for the COCO-type dataset.

    Parameters
    _________

    version : float
        version number of this dataset. Should increase every time the dataset is modified?

    description : str
        string describing the dataset

    contributor : list
        list of names of all the people contributing to the dataset

    date : datetime object
        optional date. Will use current date and time unless over-ridden.


    Returns
    ________

    infodict : dict
        the dictionary for the information that will be put into the COCO-type dataset.
    """

    if date is None:
        
        date = datetime.now()

    year = date.strftime('%Y')
    date_created = date.strftime('%Y/%m/%d %H:%M:%S')

    infoDict = { 'description': description,
                'version': version,
                'year': year,
                'contributor': contributor,
                'date_created': date_created}
    return infoDict

def datasetCategories(categories):
    """
    This function will create a dictionary that collects all of the categories 
    for the COCO-type dataset.

    Parameters
    _________

    categories : list
        List of category types


    Returns
    ________

    cata : list
        the list of dictionaries of all the categories that will be put into the COCO-type dataset.
    """

    cata = []
    for i, cat in enumerate(categories):
        cata.append({'id':i, 'name':cat})
    return cata

def datasetImages(img_dir, imagePattern, licence_id, postFix='.png'):
    """
    This function will create a dictionary that collects all of the Images 
    for the COCO-type dataset.

    Parameters
    _________

    img_dir : str
        Directory where images are located

    imagePattern : str
        string pattern to indicate that it is a screenshot image, and not a segmentation image.

    licence_id : int
        the id of the license for that image. (might be multiples, but we arent worrying about that now.)

    postFix : str
        the ending of the image. ex: .png, or .jpg, etc. This is optional. The default is .png.


    Returns
    ________

    imgList : list
        the list of dictionaries of all the images that will be put into the COCO-type dataset.
    """

    image_files = sorted(os.listdir(img_dir))
    image_files = [x for x in image_files if imagePattern in x and x.endswith(postFix)]

    imgList = []
    
    for i, img in enumerate(image_files):
        with Image.open(os.path.join(img_dir, img)) as pic:
            width, height = pic.size
        date_captured = time.ctime(os.path.getctime(os.path.join(img_dir,img)))
        
        imgList.append({"id": i, "width": width , "height": height , "file_name": img,
                    "license": licence_id, "date_captured": date_captured})
    return imgList

def annotationSingleImage(img_id, seg_mask, categories, colorMap):
    """
    This function will create a dictionary that collects all of the Images 
    for the COCO-type dataset.

    Parameters
    _________

    img_id : int
        Id of the image from which the annotations are extracted

    seg_mask : image object
        Should be an instance segmentation image

    categories : list
        List of category types

    colorMap : dict
        maps each color to the category id


    Returns
    ________

    annotation : list
        the list of dictionaries of each annotation for a single image.

    """

    annotation = []
    all_masks = create_image_mask_dictionary(seg_mask)

    id = 0
    
    for color in all_masks:

        if not color == (0, 0, 0):

            cnts = []
            area = 0
            mask = all_masks[color]
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                area += cv2.contourArea(c)
                cnts = np.concatenate(c)
            x,y,w,h = cv2.boundingRect(cnts)

            annotation.append({'id':id, 'image_id':img_id,'category_id':categories.index(colorMap[color]),
                            'bbox':[x,y,w,h], 'segmentation': np.transpose(cnts).tolist(), 'area':area, 'iscrowd':0})
            id +=1
    return annotation
   
def create_annotation_file(img_dir, seg_dir, imagePattern, segPattern, colorMap, imgpostFix='.png', segpostFix='.png', 
                            category=None, date=None, json_file=None, jsonFileName=None, version=None, description=None, 
                            contributor=None, lnames=None):
    """
    This function will create a dictionary that collects all of the Images 
    for the COCO-type dataset.

    Parameters
    _________

    img_dir : str
        full path directory of the images

    seg_dir : str
        full path directory of the segmentation files

    imagePattern : str
        string pattern to indicate that it is an screenshot image, and not a segmentation image.

    segPattern : str
        string pattern to indicate that it is an segmentation image, and not the screenshot image.

    colorMap : dict
        maps each color to the category id

    imgpostFix : str
        the ending of the real image. ex: .png, or .jpg, etc. This is optional. The default is .png.
    
    segpostFix : str
        the ending of the segmentation image. ex: .png, or .jpg, etc. This is optional. The default is .png.

    category : list
        List of category types. Optional - you dont have to over-ride it, but then you have to change
        the default below

    date : datetime object
        Optional - date. Will use current date and time unless over-ridden

    json_file : json str
        Optional - json string of information from the json file if there is one to get data from.

    jsonFileName : str
        Optional - filename for the json file if there is one to append to.

    version : float
        Optional - version number of the dataset. Default is given below

    description : str
        Optional - description of the project. Default is given below

    contributor : list
        Optional - list of contributors to the project. Default is given below

    lnames : dictionary
        the dictionary for the license for our images. 

    Returns
    ________

    finalDict : dict
        the dictionary of json data that is the original (could be empty if it wasnt provided) plus whatever is being added here.

    """

    if version is None:
        version = '1.0'
    if description is None:
        description = 'Fayetteville State University Intelligent System Laboratory: Synthetic Data for AI Detection of Rare objects'
    if contributor is None:
        contributor = ['Tivon Brown', 'Jesse Claiborne', 'Catherine Spooner','et. al.']
    if lnames is None:
        lnames = {"url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "id": 0,
            "name": "Attribution-NonCommercial-ShareAlike 4.0 International"}
            
    if category is None:
        category = ['Dome', 'Dome_Base', 'Dome2', 'Dome2_Base', 'Dome2_Bottom', 'Cone', 'Cone_Base', 'Square', 'Square_Base', 
        'SquareShade', 'SquareShade_Base', 'Triangle', 'Penta', 'Penta_Base', 'Hexa', 'Hexa_Base', 'Mini', 'Mini_Base', 
        'Mini_Hinge', 'Mini_Shade', 'Cylinder', 'Cylinder_Base', 'PTZ', 'Mic']

    if jsonFileName is None:
        jsonFileName = 'FSU_ISL_synthetic_LAS_CCTV.json'
        
    if json_file is None:
        info = datasetInfo(version, description, contributor, date)
        cate = datasetCategories(category)
        annotationList = []
    else:
        info = json_file['info']
        cate = json_file['categories']
        jsonImgList = json_file['images']
        last_idJson = jsonImgList[len(jsonImgList)-1]['id']
        annotationList = json_file['annotations']
        
    imgList = datasetImages(img_dir, imagePattern, lnames['id'], imgpostFix)
    last_idAnnotation = 0
    for i, img in enumerate(imgList):
        if json_file is not None:

            if any(img['file_name'] in d.values() for d in jsonImgList):
                print(f"The file {img['file_name']} has been added to the json. Pick a different directory.")
                return
            else:
                prevId = img['id']
                img['id'] = prevId + last_idJson
                jsonImgList.append(img)
        seg_filepath = os.path.join(seg_dir, img['file_name'].replace(imagePattern, segPattern).replace(imgpostFix, segpostFix))
        seg_mask = cv2.imread(seg_filepath)
        single_img = annotationSingleImage(img['id'], seg_mask, category, colorMap)
        if len(annotationList) > 0:
            last_idAnnotation = annotationList[len(annotationList)-1]['id']
        for anns in single_img:
            prevAnnId = anns['id']
            anns['id'] = prevAnnId + last_idAnnotation
            annotationList.append(anns)
    if json_file is not None:
        imgList = jsonImgList
    finalDict = {"info": info, "licenses":lnames, "images":imgList, "annotations":annotationList, 'categories':cate}
    
    # Serializing json
    json_object = json.dumps(finalDict, indent=2)
 
    # Writing to sample.json
    with open(jsonFileName, "w") as outfile:
        outfile.write(json_object)
        
    return(finalDict)

def main():
    #start_time = datetime.now()
    #constants

    class_types = ['Cone', 'Cylinder', 'Dome', 'Dome2', 'Square', 'SquareShade', 'Penta', 'Hexa', 'Mini', 'PTZ']
    
    for ct in class_types:
        ROOTWALKDIR = rf"F:\Files to be processed\{ct}"
        JSON_FILENAME = 'FSU_ISL_synthetic_LAS_CCTV.json'
        
        IPAT_dir = 'Img1'
        SPAT_dir = 'Seg1'
        IPAT = 'Img'
        SPAT = 'Seg'
        IMGPOSTFIX = '.png'
        SEGPOSTFIX = '.png'

        distance = ["Close", "Far"]
        
        root_walk_dirlist = os.listdir(ROOTWALKDIR)

        if not os.path.exists(JSON_FILENAME):
            print(f"{JSON_FILENAME} does not exists. It will be created.")
            json_object = None
        
        else:
            print(f"{JSON_FILENAME} exists, adding information to this file.")
            with open(JSON_FILENAME, 'r') as infile:
                json_object = json.load(infile)


        for dirs in root_walk_dirlist:
            
            dirspath = os.path.join(ROOTWALKDIR, dirs)
            alistdir = os.listdir(dirspath)
            
            rootpath = os.path.join(dirspath, alistdir[0])
        
            seg_map_file = os.path.join(rootpath, 'GTSegmentationGenerator', 'segmentation_info.json')
            colorMap = read_json(seg_map_file)      
        
            for dist in distance:
                img_dir = os.path.join(rootpath, f'{IPAT_dir}', dist)
                seg_dir = os.path.join(rootpath, f'{SPAT_dir}', dist)
                print(img_dir)
            
                annotation_dict = create_annotation_file(img_dir, seg_dir, IPAT, SPAT, colorMap, imgpostFix=IMGPOSTFIX, segpostFix=SEGPOSTFIX, json_file=json_object, jsonFileName=JSON_FILENAME)
    

if __name__ == "__main__":
    main()