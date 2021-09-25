import os
import json
import yaml
import sys
import re
import inflect
# p = inflect.engine()
import numpy as np
from tqdm import tqdm
from os import listdir, getcwd
from os.path import join
from clean_up_names import clean_objects

sys.path.append('/storage/che011/BUA/bottom-up-attention3/data/genome/')
from visual_genome_python_driver import local as vg

coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush']

'''
Gets classes list
yaml_file: includes class list under 'names'. If None is passed, default COCO classes list is used.
'''
def get_classes(yaml_file):
    if yaml_file:
        with open(yaml_file) as file:
            documents = yaml.full_load(file)
            classes = documents['names']
    else:
        classes = coco_classes
    return classes

'''
Adjust bbox coordinates to be between 0 and 1
size (tuple): (image width, image height)
box (list): original bbox [x,y,w,h] coordinates
'''
def convert(size,box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0]*dw
    y = box[1]*dh
    w = box[2]*dw
    h = box[3]*dh
    x = x + (w/2)
    y = y + (h/2)
    return (x,y,w,h)

        
'''
Removes overlapping bounding boxes
Adopted from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
boxes (2D numpy array of floats): For each line, the format is x,y,width,height
overlapThresh (float): threshold to determine which bboxes are the same item
'''
def non_max_suppression_fast(boxes, overlapThresh=0.85):
    if len(boxes) == 0:
        return []
    if len(boxes) == 1:
        return boxes
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2] - 1
    y2 = boxes[:,1] + boxes[:,3] - 1
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
    return np.unique(boxes[pick],axis=0)
    
'''
Removes overlapping items
outfile_contents can be path to label file as well
outfile_contents (list): list of strings in the format "class_id x y width height" e.g. "75 0.52625 0.15166666666666667 0.09875 0.5650000000000001"
'''
def clean_up_label(outfile_contents):
    path_input = None
    if type(outfile_contents) == str:
        path_input = outfile_contents
        with open(path_input) as f:
            outfile_contents = f.read().splitlines()
    class_dict = {}
    for line in outfile_contents:
        info = [int(x) if i == 0 else float(x) for i, x in enumerate(line.split(" "))]
        if info[0] not in class_dict.keys():
            class_dict[info[0]] = []
        class_dict[info[0]].append(info[1:])
    output = []
    for class_id, bboxes in class_dict.items():
        bboxes = non_max_suppression_fast(np.array(bboxes), overlapThresh=0.85)
        for bbox in bboxes.tolist():
            output.append(str(class_id) + " " + " ".join([str(x) for x in bbox]))
    if path_input:
        new_labels_dir = os.path.dirname(path_input).replace('labels','labels_cleaned')
        if not os.path.isdir(new_labels_dir):
            os.mkdir(new_labels_dir)
        path_output = os.path.join(new_labels_dir,os.path.basename(path_input))
        with open(path_output,'w') as f:
            f.write("\n".join(output))
    return output
    
def check_num_samples_in_file(path_input, counter):
    with open(path_input) as f:
        contents = [int(x.split(' ')[0]) for x in f.read().splitlines()]
        print(contents)
        counter['gun']+=contents.count(179)
        counter['army uniform']+= contents.count(109)
    return counter
def check_num_samples_in_dir(vg_dir):
    label_dir = os.path.join(vg_dir,'labels_cleaned')
    counter = {'gun':0, 'army uniform':0}
    for file in tqdm(os.listdir(label_dir)):
        path_input = os.path.join(label_dir,file)
        counter = check_num_samples(path_input, counter)
    print(counter)
    
'''
Removes overlapping bounding boxes in labels directory and saves it to labels_cleaned.
To be used only when this is not implemented in creation of labels files
vg_dir (str): dir that contained labels dir
'''
def clean_up_all_label_files(vg_dir):
    label_dir = os.path.join(vg_dir,'labels')
    for file in tqdm(os.listdir(label_dir)):
        path_input = os.path.join(label_dir,file)
        clean_up_label(path_input)   
    
'''
Returns plural form of word if if it singular else return None
word (str): word to get plurality form of
'''
def add_plurality(word):
    plural = p.plural_noun(word)
    if not plural:
        # singular = p.singular_noun(word)
        # return singular
        return None
    else:
        return plural

'''
Check if word is singular
word (str): word to check
'''
def is_singular(word):
    return not p.singular_noun(word)

'''
Converts labels_filtered.txt to labels_filtered.json.
labels_filtered.txt has more readability and labels_filtered.json is for easy access by system
vg_dir (str): dir that contained labels_filtered.txt
'''
def save_classes_dict_to_json(vg_dir):
    classes_dict = {}
    # p = inflect.engine()
    class_list = []
    with open(os.path.join(vg_dir,'labels_filtered.txt')) as f:
        objects_info = f.read().splitlines()
        for line in objects_info:
            line_strip = line.strip()
            if line_strip == '':
                continue
            else:
                syno_list = line_strip.split(',')
                curr_class = syno_list[0]
                class_list.append(curr_class)
                for syno in syno_list:
                    syno_strip = syno.strip()
                    if syno_strip != '':
                        classes_dict[syno_strip] = curr_class
    print('Classes: %s' % ','.join(class_list))
    print('Number of classes (nc): %d' % len(class_list))
    print('Might need to edit vg.yaml and cfg file')
    print()
    print('Cfg File changes needed are as listed below')
    print('1) Line 3: Set batch=64.')
    print('2) Line 4: Set subdivisions=32, the batch will be divided by 16 or 64 depends on GPU VRAM requirements.')
    print('3) Change line max_batches to classes*2000 but not less than number of training images, and not less than 6000, f.e. max_batches=6000 if you train for 3 classes.')
    print('4) Change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400.')
    print('5) Change line classes=2 to your number of objects in each of 3: Line 970 1058 1146.')
    print('6) Change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers. Line 1022 1131 1240')
    with open(os.path.join(vg_dir,'labels_filtered.json'), 'w') as json_f:
        json.dump(classes_dict, json_f)
        
'''
Outputs data to label directory in current dir
Each label file will correspond to each image. 
Each line in the label file will have class id, and bbox coordinates (x,y,w,h)
instance_file (str): path to instance_trainYYYY.json or instance_valYYYY.json
classes (list): list of classes to classify. Resulting label files use the index of this list as class id
'''      
def convert_annotation_coco(instance_file, classes):
    with open(instance_file,'r') as f:
        data = json.load(f)
    for item in tqdm(data['images']):
        image_id = item['id']
        file_name = item['file_name']
        width = item['width']
        height = item['height']
        value = filter(lambda item1: item1['image_id'] == image_id, data['annotations'])
        with open('/storage/che011/ICT/fairseq-image-captioning/ms-coco/labels/%s.txt'%(file_name[:-4]), 'w') as outfile:
            outfile_contents = []
            for item2 in value:
                category_id = item2['category_id']
                value1 = list(filter(lambda item3: item3['id'] == category_id, data['categories']))
                name = value1[0]['name']
                if name not in classes:
                    name = name.replace(" ","")
                class_id = classes.index(name)
                box = item2['bbox']
                bb = convert((width,height),box)
                outfile_contents.append(str(class_id)+" "+" ".join([str(a) for a in bb]))
            outfile.write('\n'.join(outfile_contents))
            
'''
Outputs data to label directory in vg_dir
Each label file will correspond to each image. 
Each line in the label file will have class id, and bbox coordinates (x,y,w,h)
vg_dir (str): path to data downloaded from http://visualgenome.org/
classes (list): list of classes to classify. Resulting label files use the index of this list as class id
'''        
def convert_annotation_vg(vg_dir, classes):
    images = vg.GetAllImageData(dataDir=vg_dir)
        
    img_dict = {img.id:img for img in images}
    sgjson = json.load(open(os.path.join(vg_dir, 'scene_graphs.json'), 'r'))
    
    with open(os.path.join(vg_dir,'labels_filtered.json')) as map_class_f:
        map_class_dict = json.load(map_class_f)
        
    image_range = images

    for i, image in tqdm(enumerate(image_range), total=len(image_range)):
        image_id = image.id
        file_name = os.path.basename(image.url)
        width = image.width
        height = image.height
        sg = vg.GetSceneGraph(image_id, img_dict, dataDir=vg_dir, sgjson=sgjson)
        outfile_contents = []
        for obj in sg.objects:
            class_id = None
            obj_name = obj.names[0]
            obj_name = clean_objects(obj_name)
            if obj_name in classes:
                class_id = classes.index(obj_name)
            elif obj_name in map_class_dict.keys():
                class_id = classes.index(map_class_dict[obj_name])
            else:
                continue
            bb = convert((width,height),(obj.x, obj.y, obj.width, obj.height))
            outfile_s = str(class_id)+" "+" ".join([str(a) for a in bb])
            if outfile_s not in outfile_contents:
                outfile_contents.append(outfile_s)
        outfile_contents = clean_up_label(outfile_contents)
        with open(os.path.join(vg_dir,'labels/%s.txt'%(file_name[:-4])), 'w') as outfile:
            outfile.write('\n'.join(outfile_contents))

'''
Remove the empty label files from the text file that contains list of image files in each split

yaml_file: includes path to 3 files that contain path to images in train, val and test splits respectively
           (e.g. /storage/che011/YOLOv4/PyTorch_YOLOv4/data/coco_vg.yaml)
'''
def clean_up_split_files(yaml_file):
    with open(yaml_file) as file:
        documents = yaml.full_load(file)
    for split in ['train','val','test']:
        print(split)
        split_file = documents[split].replace("_cleared_empty","")
        with open(split_file) as f:
            image_files = f.read().splitlines()
        label_files = []
        print('Collating label files')
        for x in tqdm(image_files):
            if x.startswith('VG'):
                im_file = os.path.join('/storage/che011/BUA/VGdata/',x.split(' ')[0])
                label_files.append(im_file.replace('VG_100K_2', 'labels').replace('VG_100K','labels').replace(os.path.splitext(im_file)[-1], '.txt'))
            elif 'val2014/' in x.strip() or 'train2014/' in x.strip():
                im_file = os.path.join('/storage/che011/ICT/fairseq-image-captioning/ms-coco/images/' ,x.split(' ')[0])
                label_files.append(im_file.replace('images', 'labels').replace(os.path.splitext(im_file)[-1], '.txt').replace('/val2014','').replace('/train2014',''))
        new_image_files = []
        new_split_file = "_cleared_empty".join(os.path.splitext(split_file))
        print('Writing new split files to: %s' % (new_split_file))
        for img, label in tqdm(zip(image_files, label_files), total=len(image_files)):
            with open(label) as f:
                contents = f.read()
                if not re.search(r'^\s*$', contents):
                    new_image_files.append(img)
        with open(new_split_file,'w') as s:
            s.write('\n'.join(new_image_files)) 
            
'''
Calls the function to generate labels for Visual Genome or COCO dataset

data_name (str): either 'coco' or 'vg'
classes (list): list of classes to classify. Resulting label files use the index of this list to refer to the respective classes
default_path (bool): indicates if the paths listed below are correct
instance_train_file (str): path to instance_trainYYYY.json. Leave blank if using default
instance_val_file (str): path to instance_valYYYY.json. Leave blank if using default
vg_dir (str): path to data downloaded from http://visualgenome.org/
'''           
def convert_annotation(data_name, classes, default_path=True, instance_train_file=None, instance_val_file=None, vg_dir=None):
    if default_path:
        instance_train_file = '/storage/che011/ICT/fairseq-image-captioning/ms-coco/annotations/instances_train2014.json'
        instance_val_file = '/storage/che011/ICT/fairseq-image-captioning/ms-coco/annotations/instances_val2014.json'
        vg_dir = '/storage/che011/BUA/VGdata/'
    if data_name == 'coco':
        if instance_train_file:
            convert_annotation_coco(instance_train_file, classes)
        if instance_val_file:
            convert_annotation_coco(instance_val_file, classes)
    elif data_name == 'vg':
        convert_annotation_vg(vg_dir, classes)
        
def run(mode):
    if mode == 'vg':
        yaml_file = '/storage/che011/YOLOv4/PyTorch_YOLOv4/data/vg.yaml'
        classes = get_classes(yaml_file)
        vg_dir = '/storage/che011/BUA/VGdata/'
        save_classes_dict_to_json(vg_dir)
        convert_annotation('vg', classes)
        # clean_up_split_files(yaml_file)
        # for i in range(1,51):
            # clean_up_label('/storage/che011/BUA/VGdata/labels_with_overlaps/%d.txt' % (i))
        clean_up_all_label_files(vg_dir)
    elif mode == 'trainvgtestcoco':
        yaml_file = '/storage/che011/YOLOv4/PyTorch_YOLOv4/data/vg.yaml'
        classes = get_classes(yaml_file)
        vg_dir = '/storage/che011/BUA/VGdata/'
        save_classes_dict_to_json(vg_dir)
        convert_annotation('coco', classes)
    else:
        print('no such dataset')
        
if __name__ == '__main__':
    run('trainvgtestcoco')
    
