# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import time
import importlib
import os
import argparse
import json
import pathlib

import torch
from torchvision import transforms
from core.utils import ZipReader

# My libs
from core.utils import Stack, ToTorchFormatTensor


parser = argparse.ArgumentParser(description="STTN")
parser.add_argument("-f", "--frame", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("-c", "--ckptpath", type=str, required=True)
parser.add_argument("-cn", "--ckptnumber", type=str, required=True)
parser.add_argument("--model", type=str, default='sttn')
parser.add_argument("--shifted", action='store_true')
parser.add_argument("--overlaid", action='store_true')
parser.add_argument("--famelimit", type=int, default=927)
parser.add_argument("--zip", action='store_true')
parser.add_argument("-g", "--gpu", type=str, default="7", required=True)
parser.add_argument("-d", "--Dil", type=int, default=8)
parser.add_argument("-r", "--readfiles", action='store_true')


args = parser.parse_args()


ref_length = 10
neighbor_stride = 5
default_fps = 24

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video 
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


# read frame-wise masks 
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        sz=m.size
        m = np.array(m.convert('L'))
        m = np.array(m > 199).astype(np.uint8) #Rema:from 0 to 199 changes to binary better
        if args.Dil !=0:
            m = cv2.dilate(m, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (args.Dil, args.Dil)), iterations=1) #Rema:Dilate only 1 iteration
        if args.shifted:
            M = np.float32([[1,0,50],[0,1,0]])
            m_T = cv2.warpAffine(m,M,sz)  
            m_T[m!=0]=0
            m = np.copy(m_T)
        masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video 
def read_frames(fpath):
    frames = []
    fnames = os.listdir(fpath)
    fnames.sort()
    for f in fnames: 
        f = Image.open(os.path.join(fpath, f))
#         f = f.resize((w, h), Image.NEAREST)
#        f = np.array(f)
#        f = np.array(f > 0).astype(np.uint8)
#        f = cv2.dilate(f, cv2.getStructuringElement(
#            cv2.MORPH_CROSS, (3, 3)), iterations=1)
        frames.append(f)
    return frames, fnames

def read_frames_mask_zip(fpath, mpath):
    frames = {}
    masks = {}
    fnames = {}
    with open(os.path.join(fpath.split("JPEGImages")[0], 'test.json'), 'r') as f:
        video_dict = json.load(f)
    video_names = list(video_dict.keys())
    for video_name in video_names: #[:1]:
        frames_v = []
        masks_v = []       
        zfilelist = ZipReader.filelist("{}/{}.zip".format(
            fpath, video_name)) #used since all_frames counts from 0 whereas zfilelist checks the correct naming of files
        fnames[video_name]=zfilelist
        for zfile in zfilelist: #[:100]:
            img = ZipReader.imread('{}/{}.zip'.format(
                fpath, video_name), zfile).convert('RGB')
            frames_v.append(img)
            m = ZipReader.imread('{}/{}.zip'.format(
                mpath, video_name), zfile).convert('RGB')
            sz=m.size
            m = np.array(m.convert('L'))
            m = np.array(m > 199).astype(np.uint8) #Rema:from 0 to 199 changes to binary better
            if args.Dil !=0:
                m = cv2.dilate(m, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (args.Dil, args.Dil)), iterations=1) #Rema:Dilate only 1 iteration change 3,3 to 55(tried it in quantifyResults.ipyb
            if args.shifted:
                M = np.float32([[1,0,50],[0,1,0]])
                m_T = cv2.warpAffine(m,M,sz)  
                m_T[m!=0]=0
                m = np.copy(m_T)
            all_mask=Image.fromarray(m*255)
            masks_v.append(all_mask)
        frames[video_name]=frames_v
        masks[video_name]=masks_v
        print(video_name)
        
    return frames, fnames, masks, video_names, sz

def evaluate(w, h, frames, fnames, masks, video_name, model, device, overlaid, shifted, Dil):
    #added for memory issue
    if len(frames)>args.famelimit:
        masks=masks[:args.famelimit]
        frames=frames[:args.famelimit]
  
    video_length = len(frames)
    feats = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]

        
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    feats, masks = feats.to(device), masks.to(device)
    comp_frames = [None]*video_length
    
    with torch.no_grad():
        feats = model.encoder((feats*(1-masks).float()).view(video_length,3, h, w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)
    print('loading frames and masks from: {}'.format(args.frame))

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(neighbor_ids, video_length)
        with torch.no_grad():
            print(feats.shape)
            pred_feat = model.infer(
                feats[0, neighbor_ids+ref_ids, :, :, :], masks[0, neighbor_ids+ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(
                pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                if args.overlaid:
                    overlay_mult=binary_masks[idx]
                    overlay_add=frames[idx] * (1-binary_masks[idx])
                else:
                    overlay_mult=1
                    overlay_add=0
                img = np.array(pred_img[i]).astype(
                    np.uint8)*overlay_mult+overlay_add
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5
                    #Rema:
    savebasepath=os.path.join(args.output,"gen_"+args.ckptnumber.zfill(5),"full_video",video_name, overlaid, shifted, Dil)
    frameresultpath=os.path.join(savebasepath,"frameresult")
    pathlib.Path(frameresultpath).mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(savebasepath+"/result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
    for f in range(video_length):
        if args.overlaid:
            overlay_mult=binary_masks[f]
            overlay_add=frames[f] * (1-binary_masks[f])
        else:
            overlay_mult=1
            overlay_add=0
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*overlay_mult+overlay_add
        fnameNew=os.path.basename(fnames[f])
        cv2.imwrite(frameresultpath+f"/{fnameNew}",cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(savebasepath+"/result.mp4"))

def main_worker():
    overlaid="overlaid" if args.overlaid else "notoverlaid"
    shifted="shifted" if args.shifted else "notshifted"
    Dil = "noDil" if args.Dil == 0 else ""
    # set up models 
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    

    model = net.InpaintGenerator().to(device)
    model_path = os.path.join(args.ckptpath,"gen_"+args.ckptnumber.zfill(5)+".pth")
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.ckptpath))
    model.eval()
    
    if args.zip:
        file1 = os.path.join(args.frame.split("JPEGImages")[0], 'files/testframes_v.npy') # 'files/frames_v.npy')
        file2 = os.path.join(args.frame.split("JPEGImages")[0], 'files/testfnames_v.npy') # 'files/fnames_v.npy')
        file3 = os.path.join(args.frame.split("JPEGImages")[0], 'files/testmasks_v.npy') # 'files/masks_v.npy')
        file4 = os.path.join(args.frame.split("JPEGImages")[0], 'files/testvideo_names.npy') # 'files/video_names.npy')
        file5 = os.path.join(args.frame.split("JPEGImages")[0], 'files/testsz.npy') # 'files/sz.npy')
        file1Ex = os.path.isfile(file1)
        file2Ex = os.path.isfile(file2)
        file3Ex = os.path.isfile(file3)
        file4Ex = os.path.isfile(file4)
        file5Ex = os.path.isfile(file5)

        if file1Ex and file2Ex and file3Ex and file4Ex and file5Ex and args.readfiles:
            # start timer
            start = time.time()
            frames_v = np.load(file1, allow_pickle='TRUE').item()
            # end timer
            end = time.time()
            print("frames_v loaded")
            print(f"Time taken to load frames_v: {end - start} seconds") 
            fnames_v = np.load(file2, allow_pickle='TRUE').item()
            print("fnames_v loaded")
            masks_v = np.load(file3, allow_pickle='TRUE').item()
            print("masks_v loaded")
            video_names = np.load(file4, allow_pickle='TRUE')
            print("video_names loaded")
            sz = np.load(file5, allow_pickle='TRUE')
            print("sz loaded")
            print("files loaded...")
        else:
            os.makedirs(os.path.join(args.frame.split("JPEGImages")[0], 'files'), exist_ok=True)
            frames_v, fnames_v, masks_v, video_names, sz = read_frames_mask_zip(args.frame, args.mask)
            np.save(file1, frames_v) 
            np.save(file2, fnames_v) 
            np.save(file3, masks_v) 
            np.save(file4, video_names) 
            np.save(file5, sz) 

        w, h = sz
        for video_name in video_names:
            frames = frames_v[video_name]
            fnames = fnames_v[video_name]
            masks = masks_v[video_name]
            evaluate(w, h, frames, fnames, masks, video_name, model, device, overlaid, shifted, Dil)
    else:
        # prepare datset, encode all frames into deep space 
        video_name=os.path.basename(args.frame.rstrip("/"))
        frames, fnames = read_frames(args.frame)
        w, h=frames[0].size
        masks = read_mask(args.mask)
        evaluate(w, h, frames, fnames, masks, video_name, model, device, overlaid, shifted, Dil)
     
  
if __name__ == '__main__':
    main_worker()
