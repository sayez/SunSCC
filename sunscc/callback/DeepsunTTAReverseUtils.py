import torch
from kornia.geometry.transform import warp_affine, get_rotation_matrix2d
import numpy as np

def reverse_batch_ShiftScaleRotate(img, params):
    angle = -params['angle'].double().cpu()
    scale = (1/params['scale'].double()).unsqueeze(1).repeat(1,2).cpu()
    # print(scale)
    # scale = 1/params['scale'].double()
    dx = params['dx'].double().cpu()
    dy = params['dy'].double().cpu()

    # print(angle, scale) 
    # width,height = img.shape[1], img.shape[2]
    # # define the rotation center
    # center = torch.ones((img.shape[0], 2)).double().cpu()
    # center[..., 0] = (dx* width) + img.shape[2] / 2  # x
    # center[..., 1] = (dy* height) + img.shape[1] / 2  # y   


    width,height = params['cols'].double().cpu(), params['rows'].double().cpu()

    center = torch.ones((img.shape[0], 2)).double().cpu()
    center[..., 0] = (dx* width) + width / 2  # x
    center[..., 1] = (dy* height) + height / 2  # y   
    
    # print( dx.device,dy.device, angle.device, scale.device, center.device)

    # compute the transformation matrix
    M = get_rotation_matrix2d(center, angle, scale)
    # print(M.shape)

    M[:,0, 2] -=  dx * width
    M[:,1, 2] -=  dy * height

    # apply the transformation to original image
    _, h, w = img.shape
    img_warped = warp_affine(img.cpu().unsqueeze(1).double()+1, M, dsize=(h, w), flags='nearest')-1
    # img_warped = warp_affine(img.unsqueeze(1).double(), M, dsize=(h, w), flags='nearest')
    img_warped = img_warped.squeeze(1)

    return img_warped



def reverse_img_Rotate90(img, factor):
    # print('there', img.shape, factor )
    unrotated_img = torch.rot90(img, factor)
    return unrotated_img

def reverse_batch_Rotate90(img_batch, params):
    # print('here', img_batch.shape, params['factor'].shape)
    out_batch = torch.zeros_like(img_batch)
    for i in range(img_batch.shape[0]):
        out_batch[i] = reverse_img_Rotate90(img_batch[i], -params['factor'][i])
    # print(out_batch.shape)
    return out_batch
    
