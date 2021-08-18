# Merges train, val, test splits from karpathy on coco and botttom up attention on visual genome

train_val_test = {'train': ['/storage/che011/ICT/fairseq-image-captioning/splits/karpathy_train_images.txt', '/storage/che011/BUA/bottom-up-attention3/data/genome/train.txt'],
                    'val': ['/storage/che011/ICT/fairseq-image-captioning/splits/karpathy_valid_images.txt', '/storage/che011/BUA/bottom-up-attention3/data/genome/val.txt'],
                   'test': ['/storage/che011/ICT/fairseq-image-captioning/splits/karpathy_test_images.txt', '/storage/che011/BUA/bottom-up-attention3/data/genome/test.txt']}
               
output_paths = {'train': '/storage/che011/YOLOv4/PyTorch_YOLOv4/data/coco_vg_train.txt',
                'val': '/storage/che011/YOLOv4/PyTorch_YOLOv4/data/coco_vg_val.txt',
                'test': '/storage/che011/YOLOv4/PyTorch_YOLOv4/data/coco_vg_test.txt'}
               
def merge(path_list, output_path):
    content = []
    for path in path_list:
        with open(path) as f:
            content.extend(f.read().splitlines())
    with open(output_path,'w') as output:
        output.write('\n'.join(content))
    
def merge_coco_vg_trainvaltest(train_val_test, output_paths):
    for key, value in train_val_test.items():
        output_path = output_paths[key]
        merge(value, output_path)
        print("%s is done and saved to %s" % (key, output_path))
        
merge_coco_vg_trainvaltest(train_val_test, output_paths)
            
    
            