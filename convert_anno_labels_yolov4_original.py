import os
import json
import yaml
import sys
from tqdm import tqdm
from os import listdir, getcwd
from os.path import join

import nltk
nltk.download('wordnet')

sys.path.append('/storage/che011/BUA/bottom-up-attention3/data/genome/')
from visual_genome_python_driver import local as vg
from nltk.corpus import wordnet as wn

animals = [x.name().split('.')[0] for x in wn.synsets('animal')[0].closure(lambda x:x.hyponyms())]
artifact = wn.synset('artifact.n.01')
natural_object = wn.synset('natural_object.n.01')

not_dict = {
    'person' : ['printer', 'drawer', 'folder', 'self', 'holder', 'speaker', 'washer', 'calculator', 'lookout', 
            'plater', 'broad', 'knocker', 'tier', 'switcher', 'shaker', 'macaroni', 'spearhead', 'puncher', 
            'general', 'processor', 'coaster', 'natural', 'sprayer', 'beast', 'hauler', 'manikin', 'double', 
            'forward', 'sealer', 'batter', 'great', 'pussycat', 'manufacturer', 'religious', 'tackle', 'ideal',
            'spoiler', 'soul', 'breaker', 'grader', 'cutter', 'cavalier', 'bowler', 'maroon', 'ma', 'porter',
            'wanton', 'topper', 'silly', 'mandarin', 'wiper', 'transfer', 'rake', 'butter', 'metropolitan', 
            'tipper', 'heavy', 'flier', 'modern', 'spinner', 'authority', 'sovereign', 'excavator', 'mailer',
            'broadcaster', 'catcher', 'stringer', 'builder', 'chink', 'musher', 'browser', 'tracker', 'dodger',
            'contemplative', 'distiller', 'decoder', 'hunk', 'trader', 'mannequin', 'saver', 'riser', 'dick',
            'best', 'hag', 'bishop', 'peeler', 'fixer', 'cookie', 'sage', 'baster', 'beater', 'leveler', 'sneak',
            'fastener', 'ward', 'rider', 'waterer', 'guide', 'sledder', 'loader', 'cog', 'conveyer', 'trusty', 
            'escalader', 'pusher', 'putter', 'shaver', 'dummy', 'invalid', 'coach', 'diver', 'host', 'canon', 
            'choker', 'toast', 'pivot', 'date', 'breeder', 'striper', 'bard', 'clapper', 'cobbler', 'rocker', 
            'bleacher', 'anti', 'sorter', 'quack', 'plier', 'descendant', 'router', 'conditioner', 'roaster', 
            'scratcher', 'scrubber', 'national', 'cardholder', 'striker', 'closer', 'slicer', 'ringer', 'preserver',
            'ruler', 'bore', 'waver', 'hood', 'reader', 'flop', 'hummer', 'carrier', 'grabber', 'scanner', 'wally',
            'ancient', 'jumper', 'marine', 'member', 'mover', 'ensign', 'tumbler', 'nibbler', 'blocker', 'dry', 
            'powerhouse', 'matador', 'winder', 'remover', 'finder', 'hotdog', 'creature', 'trace', 'server', 'wisp',
            'smoothie', 'blond', 'sipper', 'changer', 'sphinx', 'divider', 'bouncer', 'pitcher', 'hanger', 'digger'],
    'car' : ['machine'],
    'bicycle' : ['wheel', 'bike'],
    'window': ['skylight','transom'],
    'horse': ['charger', 'cob'],
    'jeans': ['denim'],
    'toilet': ['can'],
    'night': ['dark'],
    'shop': ['garage'],
    'engine': ['windmill', 'diesel'],
    'cake': ['bar'],
    'field': ['campus'],
    'building': ['aviary', 'dollhouse', 'barn', 'garage', 'restaurant', 'cabin', 'theater', 'shed', 'casino', 'rink', 
                'brownstone', 'carport', 'cafe', 'outhouse', 'lunchroom', 'tavern', 'ruin', 'boathouse','multiplex', 'stable', 
                'kennel', 'greenhouse'],
    'room': ['rotunda', 'court', 'compartment', 'belfry'],
    'pants': ['drawers'],
    'platform': ['boards', 'pallet'], 
    'suitcase': ['grip'], 
    'cup': ['beaker'],
    'handbag': ['pocketbook'],
    'flag': ['colors'],
    'runway': ['rail','railway','rails', 'track'],
    'sandwich': ['western', 'gyro'],
    'bridge': ['gangplank'],
    'day': ['birthday'],
    'pens': ['quill'],
    'bouquet': ['corsage'],
    'trolley': ['tram', 'streetcar'],
    'valley': ['hollow', 'draw']
            }

def get_classes(yaml_file):
    if yaml_file:
        with open(yaml_file) as file:
            documents = yaml.full_load(file)
            classes = documents['names']
    else:
        classes = ["person","bicycle","car","motorcycle","airplane","bus","train",
               "truck","boat","traffic light","fire hydrant","stop sign","parking meter",
               "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
               "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
               "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
               "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
               "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
               "donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv",
               "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
               "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
    return classes
    
def convert_str_dict_get_iou(string):
    info = [int(x) for x in string.split(" ")]
    return {'x':info[0],'y':info[1],'width':info[2],'height':info[3]}
    
def get_iou(bb1, bb2):
    
    if type(bb1) == str and type(bb2) == str:
        bb1 = convert_str_dict_get_iou(bb1)
        bb2 = convert_str_dict_get_iou(bb2)

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x'], bb2['x'])
    y_top = max(bb1['y'], bb2['y'])
    x_right = min(bb1['x']+bb1['width'], bb2['x']+bb2['width'])
    y_bottom = min(bb1['y']+bb1['height'], bb2['y']+bb2['height'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = bb1['width'] * bb1['height']
    bb2_area = bb2['width'] * bb2['height']

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

#box form[x,y,w,h]
def convert(size,box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0]*dw
    y = box[1]*dh
    w = box[2]*dw
    h = box[3]*dh
    return (x,y,w,h)
    
def get_image_list():
    list_dir = {'train' : '/storage/che011/ICT/fairseq-image-captioning/splits/karpathy_train_images.txt',
                'valid' : '/storage/che011/ICT/fairseq-image-captioning/splits/karpathy_valid_images.txt',
                'test' : '/storage/che011/ICT/fairseq-image-captioning/splits/karpathy_test_images.txt'}
    for img_list_dir in list_dir:
        assert os.path.isfile(img_list_dir)
    with open(list_dir['train']) as train_listfile, open(list_dir['valid']) as valid_listfile, open(list_dir['test']) as test_listfile:
        train_val_test = {'train': [int(x.split(' ')[1]) for x in train_listfile.read().splitlines()],
                            'valid': [int(x.split(' ')[1]) for x in valid_listfile.read().splitlines()],
                            'test': [int(x.split(' ')[1]) for x in test_listfile.read().splitlines()]}
    return train_val_test

def add_synohyponyms(word):
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(word)
    typesOfWord = []
    for synset in synsets:
        typesOfWord.extend(synset.lemma_names())
        typesOfWord.extend(set([w for s in synset.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
    typesOfWord = set(typesOfWord)
    return typesOfWord
    
def add_less_synohyponyms(word, path_threshold = 0, meaning_threshold = 2):

    if word == 'room':
        path_threshold = 0.5
        
    if word == 'person':
        with open('/storage/che011/BUA/VGdata/objects_vocab_person.txt') as person_vocab:
            pv = [x.split(',')[0] for x in person_vocab.read().splitlines()]
            return pv

    synsets = wn.synsets(word)
    if len(synsets) == 0 or word == 'group':
        return []
    else:
        synset = synsets[0]
    typesOfWord = []
    typesOfWord.extend(synset.lemma_names())
    hyponyms = []
    for s in synset.closure(lambda s:s.hyponyms()):
        s_name = s.name().split('.')[0]
        if word == 'person' and s_name in animals:
            continue
        if word == 'person' and natural_object in  wn.synsets(s_name)[0].closure(lambda s:s.hypernyms()):
            continue
        if word == 'person' and artifact in  wn.synsets(s_name)[0].closure(lambda s:s.hypernyms()):
            continue
        if synset.path_similarity(s) >= path_threshold and int(s.name().split('.')[-1]) <= meaning_threshold:
            # print(s.name(), int(s.name().split('.')[-1]), synset.path_similarity(s), s.lemma_names()[0])
            hyponyms.append(s.lemma_names()[0])
    if word == 'tv':
        hyponyms.append('television')
    hyponyms = set(hyponyms)
    typesOfWord.extend(hyponyms)
    typesOfWord = set(typesOfWord)
    return typesOfWord
    
def add_plurality(word):
    import inflect
    p = inflect.engine()
    plural = p.plural_noun(word)
    if not plural:
        # singular = p.singular_noun(word)
        # return singular
        return None
    else:
        return plural
    
def create_big_classes(classes):
    classes_big = []
    classes_index = []
    print("Creating Big Classes")
    for c in tqdm(classes):
        idx = classes.index(c)
        classes_big.append(c)
        classes_index.append(idx)
        plural = add_plurality(c)
        if plural:
            classes_big.append(add_plurality(c))
        classes_index.append(idx)
        if True:
            synohyponyms = [x for x in add_less_synohyponyms(c) if x not in classes]
            plurality_temp = [add_plurality(word) for word in synohyponyms]
            plurality = list(filter(None, plurality_temp))
            classes_big.extend(synohyponyms)
            classes_index.extend([idx for i in range(len(synohyponyms))])
            # classes_big.extend(plurality)
            # classes_index.extend([classes.index('group') for i in range(len(plurality))])
    return classes_big, classes_index
    
def save_classes_big(classes_big, classes_index, classes, save_dir):
    with open(save_dir,'w') as save:
        save.write('\n'.join([x.replace('_', ' ')+','+classes[y] for x, y in zip(classes_big, classes_index)]))
    print("classes file saved to: ", save_dir)
    
def convert_annotation_coco(instance_file, classes):
    with open(instance_file,'r') as f:
        data = json.load(f)
    for item in tqdm(data['images']):
        image_id = item['id']
        file_name = item['file_name']
        width = item['width']
        height = item['height']
        value = filter(lambda item1: item1['image_id'] == image_id, data['annotations'])
        with open('labels/%s.txt'%(file_name[:-4]), 'w') as outfile:
            outfile_contents = []
            for item2 in value:
                category_id = item2['category_id']
                value1 = list(filter(lambda item3: item3['id'] == category_id, data['categories']))
                name = value1[0]['name']
                class_id = classes.index(name)
                box = item2['bbox']
                bb = convert((width,height),box)
                outfile_contents.append(str(class_id)+" "+" ".join([str(a) for a in bb]))
            outfile.write('\n'.join(outfile_contents))
            
def convert_annotation_vg(vg_dir, classes, save_dir, generate_classes_big=False):
    images = vg.GetAllImageData(dataDir=vg_dir)
    classes_big_used = {}
    try:
        written = int(os.listdir(os.path.join(vg_dir,'labels'))[-1].replace('.txt',''))
        image_range = images[written:]
    except IndexError:
        image_range = images
        
    image_range = images
        
    img_dict = {img.id:img for img in images}
    sgjson = json.load(open(os.path.join(vg_dir, 'scene_graphs.json'), 'r'))
    if generate_classes_big:
        classes_big, classes_index = create_big_classes(classes)
        save_classes_big(classes_big, classes_index, classes, save_dir)
    else:
        with open(save_dir) as s:
            classes_big = []
            classes_index = []
            for class_info in s.read().splitlines():
                c_big, c = class_info.split(',')
                classes_big.append(c_big)
                classes_index.append(classes.index(c))
    assert len(classes_big) == len(classes_index)
    for i, image in tqdm(enumerate(image_range)):
        image_id = image.id
        file_name = os.path.basename(image.url)
        width = image.width
        height = image.height
        sg = vg.GetSceneGraph(image_id, img_dict, dataDir=vg_dir, sgjson=sgjson)
        outfile_contents = []
        for obj in sg.objects:
            name = None
            class_id = None
            for obj_name in obj.names:
                if obj_name in classes:
                    class_id = classes.index(obj_name)
                if obj_name in classes_big:
                    name = obj_name
                    break
            if not name and not class_id:
                continue
            if not class_id:
                class_id = classes_index[classes_big.index(name)]
            if name and (classes[class_id] in not_dict.keys()) and (name in not_dict[classes[class_id]]):
                continue
            if classes[class_id] not in classes_big_used.keys():
                classes_big_used[classes[class_id]] = []
            classes_big_used[classes[class_id]].append(name)
            bb = convert((width,height),(obj.x, obj.y, obj.width, obj.height))
            outfile_contents.append(str(class_id)+" "+" ".join([str(a) for a in bb]))
        with open(os.path.join(vg_dir,'labels/%s.txt'%(file_name[:-4])), 'w') as outfile:
            outfile.write('\n'.join(outfile_contents))
        if i%10000==0:
            save_dir_temp = os.path.splitext(save_dir)
            class_big_used_str = ''
            with open(save_dir_temp[0]+'_used'+save_dir_temp[1],'w') as save_out:
                for key, value in classes_big_used.items():
                    class_big_used_str = class_big_used_str + 'CLASS: ' + key + '\n' + '\n'.join(set(value))+'\n\n'
                save_out.write(class_big_used_str)
                    
def convert_annotation(data_name, classes, default_path=True, instance_train_file=None, instance_val_file=None, vg_dir=None, save_dir=None):
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
        convert_annotation_vg(vg_dir, classes, save_dir, generate_classes_big=True)
			
if __name__ == '__main__':
    # typesOfPerson = add_less_synohyponyms('room')
    # print('box' in typesOfPerson)
    # exit()
    yaml_file = '/storage/che011/YOLOv4/PyTorch_YOLOv4/data/vg.yaml'
    classes = get_classes(yaml_file)
    vg_dir = '/storage/che011/BUA/VGdata/'
    save_dir = os.path.join(vg_dir, 'objects_vocab.txt')
    convert_annotation('vg', classes, vg_dir=vg_dir, save_dir=save_dir)