from __future__ import print_function
import click
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
parser.add_argument('--input', type=str, required=False, default=f'{cwd}/dataset/BSDS300/Sources/', help='input folder to use')
parser.add_argument('--model', type=str, default='model_trained/dbpn_x2.pth', help='model file to use')
parser.add_argument('--output', type=str, default=f'{cwd}/dataset/BSDS300/Results/', help='where to save the output images')
args = parser.parse_args()
print(args)


GPU_IN_USE = torch.cuda.is_available()

# ===========================================================
# model import & setting
# ===========================================================
device = torch.device('cuda' if GPU_IN_USE else 'cpu')
modelStateDict = torch.load(args.model, map_location=lambda storage, loc: storage)

# update names to circumvent bug; don't know why
newStateDict = OrderedDict()
for k, v in modelStateDict.items():
    name = k[7:]
    newStateDict[name] = v

model = DBPN(num_channels=3, base_channels=64, feat_channels=256, num_stages=7, scale_factor=2)
model.load_state_dict(newStateDict)
model.to(device)
model.eval()

if GPU_IN_USE:
    cudnn.benchmark = True

threadMax = multiprocessing.cpu_count()
torch.set_num_threads(threadMax)

modelFilename = os.path.basename(args.model)
modelFilename, modelFileExtension = os.path.splitext(modelFilename)

image_path_list = []
for root, subdirs, files in os.walk(args.input):
    for filename in files:
        filenameLowerCase = filename.lower()
        if (filenameLowerCase.endswith('.png') or filenameLowerCase.endswith('.jpg')):
            input_path = os.path.join(args.input, filename)

            name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output, f"{name}_super-resolve_{modelFilename}{ext}")

            image_path_list.append([input_path, output_path])

print("Model: %s" % modelFilename)
print('Found %d images' % (len(image_path_list)))

with click.progressbar(image_path_list) as bar:
    # iterate
    for input_path, output_path in bar:
        print('- %s -> %s' % (os.path.basename(input_path), os.path.basename(output_path)))

        # ===========================================================
        # input image setting
        # ===========================================================
        img = Image.open(input_path).convert('YCbCr')
        # y, cb, cr = img.split()
        data = (ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
        data = data.to(device)

        # ===========================================================
        # output and save image
        # ===========================================================
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

        out_img.save(output_path)
