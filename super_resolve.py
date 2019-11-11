from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import multiprocessing
from PIL import Image
from torchvision.transforms import ToTensor
from DBPN.model import DBPN, DBPNLL
from collections import OrderedDict

import numpy as np

# ===========================================================
# Argument settings
# ===========================================================
cwd = os.getcwd()
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input', type=str, required=False, default=f'{cwd}/dataset/BSDS300/images/val/3096.png', help='input image to use')
parser.add_argument('--model', type=str, default='model_path.pth', help='model file to use')
parser.add_argument('--output', type=str, default='test.jpg', help='where to save the output image')
args = parser.parse_args()
print(args)


# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
img = Image.open(args.input).convert('YCbCr')
y, cb, cr = img.split()


# ===========================================================
# model import & setting
# ===========================================================
device = torch.device('cuda' if GPU_IN_USE else 'cpu')
modelStateDict = torch.load(args.model, map_location=lambda storage, loc: storage)

newStateDict = OrderedDict()
for k, v in modelStateDict.items():
    name = k[7:]
    newStateDict[name] = v

model = DBPN(num_channels=3, base_channels=64, feat_channels=256, num_stages=7, scale_factor=2)
model.load_state_dict(newStateDict)
model.to(device)
model.eval()

# data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
data = (ToTensor()(img)).view(1, -1, y.size[1], y.size[0])
data = data.to(device)

if GPU_IN_USE:
    cudnn.benchmark = True


# ===========================================================
# output and save image
# ===========================================================
threadMax = multiprocessing.cpu_count()
torch.set_num_threads(threadMax)
out = model(data)
if False:
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
else:
    out_img = out.data[0].numpy()
    out_img *= 255.0
    out_img = np.uint8(out_img)
    out_img = np.moveaxis(out_img, 0, -1)
    out_img = Image.fromarray(out_img, mode="YCbCr").convert("RGB")

out_img.save(args.output)
print('output image saved to ', args.output)
