import torch
import torchextractor as tx
from models.models import *
from utils.datasets import create_dataloader
from utils.torch_utils import select_device

# Define parameters
data = 'data/vg.yaml'
cfg = 'cfg/yolov4-pacsp_vg.cfg'
task = 'test'
device= '0'
batch_size=16
weights = '/storage/che011/YOLOv4/PyTorch_YOLOv4/runs/exp0_yolov4-pacsp_vg/weights/best.pt'
imgsz=640

# Select device
device = select_device(device, batch_size=batch_size)

# Define model
model = Darknet(cfg).to(device)  

# Load Checkpoint
try:
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
except:
    load_darknet_weights(model, weights)
    
# Check img_size
imgsz = check_img_size(imgsz, s=32)  

# Half
half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()

# Configure
model.eval()
with open(data) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
nc = int(data['nc'])  # number of classes

for name, layer in model.named_modules():
    print(name, layer)
model = tx.Extractor(model, ["module_list.173.Conv2d", "module_list.158.Conv2d", "module_list.143.Conv2d"])

# Dataloader
img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
path = data['test'] if task == 'test' else data['val']  # path to val/test images
print("Data path: ", path)
dataloader = create_dataloader(path, imgsz, batch_size, 32, opt,
                               hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]


for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
    img = img.to(device, non_blocking=True)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    targets = targets.to(device)
    nb, _, height, width = img.shape  # batch size, channels, height, width
    whwh = torch.Tensor([width, height, width, height]).to(device)

    # Disable gradients
    with torch.no_grad():
        # Run model
        model_output, features = model(img, augment=augment)  # inference and training outputs
        inf_out, train_out = model_output
        print(features)
        exit()

