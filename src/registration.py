# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 13:19:07 2021

@author: shaon
"""
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import cv2

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def padding_zeros_2D(img, shape):
    
    if len(np.shape(img)) == 2:
        padded = np.zeros((np.shape(img)[0] + shape[0], np.shape(img)[1] + shape[1]))
        padded[shape[0]/2:-shape[0]/2, shape[1]/2:-shape[1]/2] = img
    elif len(np.shape(img)) == 3:
        padded = np.zeros((np.shape(img)[0] + shape[0], np.shape(img)[1] + shape[1], np.shape(img)[2]))
        padded[shape[0]/2:-shape[0]/2, shape[1]/2:-shape[1]/2, :] = img
        
    return padded

def preprocess():
    # read MR image slice
    MR = 'D:/Job/MR-Pathology/MR/aaa0043/09-16-2000-PELVISPROSTATE-50407/4.000000-T2AXIALSMFOV-51544/T2.nii'
    MR_LABEL = 'D:/Job/MR-Pathology/MR/aaa0043/09-16-2000-PELVISPROSTATE-50407/4.000000-T2AXIALSMFOV-51544/T2_label.nii'
    mr_img = sitk.ReadImage(MR)
    mr_label_img = sitk.ReadImage(MR_LABEL)
    
    mr_img_data = sitk.GetArrayFromImage(mr_img)
    mr_label_img_data = sitk.GetArrayFromImage(mr_label_img)
    
    mr_image_data_slice = mr_img_data[12,::-1,]
    mr_label_img_data_slice = mr_label_img_data[12,:,:]
    
    mr_masked = mr_image_data_slice * mr_label_img_data_slice
    index = np.where(mr_label_img_data_slice == 1)
    mr_masked = mr_masked[index[0].min()-5 : index[0].max()+10, index[1].min()-5 : index[1].max()+10]
    mr_masked = np.uint8(mr_masked.astype(np.float)/np.max(mr_masked)*255)
    mr_masked = cv2.resize(mr_masked, (128,128), interpolation=cv2.INTER_LINEAR)
    plt.imsave('./MR_masked.png', mr_masked, cmap='gray')
    mr_masked = sitk.GetImageFromArray(mr_masked)
    sitk.WriteImage(mr_masked, './MR_masked.nii.gz')
    img = np.uint8(mr_label_img_data_slice.astype(np.float)/np.max(mr_label_img_data_slice)*255)
    img = img[index[0].min()-5 : index[0].max()+5, index[1].min()-5 : index[1].max()+5]
    
    img = cv2.resize(img, (128,128), interpolation=cv2.INTER_NEAREST)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, './MR_label.nii.gz')
    
    
    # read Pathology image
    Pathology = 'D:/Job/MR-Pathology/Pathology/aaa0043D/img.png'
    Pathology_label = 'D:/Job/MR-Pathology/Pathology/aaa0043D/label.png'
    Pathology_ori = plt.imread(Pathology)
    Pathology_data = rgb2gray(plt.imread(Pathology))
    Pathology_label_data = rgb2gray(plt.imread(Pathology_label))
    
    Pathology_masked = Pathology_data * Pathology_label_data
    # padding by zeros
    '''
    mask = np.zeros((np.shape(Pathology_masked)[0]+200, np.shape(Pathology_masked)[1]+200))
    mask[:-200,:-200] = Pathology_masked
    Pathology_masked = mask
    '''
    Pathology_masked = padding_zeros_2D(Pathology_masked, (600, 600))
    
    Pathology_masked = cv2.resize(Pathology_masked, (128,128), interpolation=cv2.INTER_LINEAR)
    Pathology_label_data = cv2.resize(Pathology_label_data, (128,128), interpolation=cv2.INTER_NEAREST)
    
    plt.imsave('./Pathology_masked.png', Pathology_masked, cmap='gray')
    Pathology_masked = sitk.GetImageFromArray(np.uint8(Pathology_masked.astype(np.float)/np.max(Pathology_masked)*255))
    Pathology_masked.CopyInformation(mr_masked)
    sitk.WriteImage(Pathology_masked, './Pathology_masked.nii.gz')
    p_img = sitk.GetImageFromArray(np.uint8(Pathology_label_data.astype(np.float)/np.max(Pathology_label_data)*255))
    p_img.CopyInformation(img)
    sitk.WriteImage(p_img, './Pathology_label.nii.gz')
    
if __name__ == '__main__':
    preprocess()