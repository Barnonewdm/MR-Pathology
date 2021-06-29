# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 13:19:07 2021

@author: shaon
"""
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import cv2
import sys
import os

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def padding_zeros_2D(img, shape):
    
    if len(np.shape(img)) == 2:
        padded = np.zeros((np.shape(img)[0] + shape[0], np.shape(img)[1] + shape[1]))
        padded[np.int(shape[0]/2):-np.int(shape[0]/2), np.int(shape[1]/2):-np.int(shape[1]/2)] = img
    elif len(np.shape(img)) == 3:
        padded = np.zeros((np.shape(img)[0] + shape[0], np.shape(img)[1] + shape[1], np.shape(img)[2]))
        padded[np.int(shape[0]/2):-np.int(shape[0]/2), np.int(shape[1]/2):-np.int(shape[1]/2), :] = img
        
    return padded

def preprocess_1(img, padding=(600,600), output_dims=(256,256), interpolation='linear'):
    img = padding_zeros_2D(img, padding)
    if interpolation == 'linear':
        img = cv2.resize(img, output_dims, interpolation=cv2.INTER_LINEAR)
    elif interpolation == 'nearest':
        img = cv2.resize(img, output_dims, interpolation=cv2.INTER_NEAREST)
        
    return img
    

def preprocess(output_dims=(256,256)):
    # read MR image slice
    MR = '../MR/aaa0043/09-16-2000-PELVISPROSTATE-50407/4.000000-T2AXIALSMFOV-51544/T2.nii'
    MR_LABEL = '../MR/aaa0043/09-16-2000-PELVISPROSTATE-50407/4.000000-T2AXIALSMFOV-51544/T2_label.nii'
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
    mr_masked = cv2.resize(mr_masked, output_dims, interpolation=cv2.INTER_LINEAR)
    plt.imsave('./MR_masked.png', mr_masked, cmap='gray')
    mr_masked = sitk.GetImageFromArray(mr_masked)
    sitk.WriteImage(mr_masked, './MR_masked.nii.gz')
    img = np.uint8(mr_label_img_data_slice.astype(np.float)/np.max(mr_label_img_data_slice)*255)
    img = img[index[0].min()-5 : index[0].max()+5, index[1].min()-5 : index[1].max()+5]
    
    img = cv2.resize(img, output_dims, interpolation=cv2.INTER_NEAREST)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, './MR_label.nii.gz')
    
    
    # read Pathology image
    Pathology = '../Pathology/aaa0043D/img.png'
    Pathology_label = '../Pathology/aaa0043D/label.png'
    Pathology_ori = plt.imread(Pathology)
    #Pathology_ori = padding_zeros_2D(Pathology_ori, (600, 600))
    #Pathology_ori = cv2.resize(Pathology_ori, output_dims, interpolation=cv2.INTER_LINEAR)
    Pathology_ori = preprocess_1(Pathology_ori)
    Pathology_data = rgb2gray(plt.imread(Pathology))
    Pathology_label_data = rgb2gray(plt.imread(Pathology_label))
    Pathology_label_data[Pathology_label_data < Pathology_label_data.max()] = 0
    Pathology_label_data[Pathology_label_data >0] = 1
    
    Pathology_masked = Pathology_data * Pathology_label_data
    Pathology_masked = preprocess_1(Pathology_masked)

    Pathology_label_data = preprocess_1(Pathology_label_data, interpolation='nearest')
    
    for i in range(np.shape(Pathology_ori)[2]):
        P_ori_img = sitk.GetImageFromArray(Pathology_ori[:,:,i] * Pathology_label_data)
        #P_ori_img.CopyInformation(img)
        sitk.WriteImage(P_ori_img, './Pathology_ori_' + str(i) + '.nii.gz')
    
    plt.imsave('./Pathology_masked.png', Pathology_masked, cmap='gray')
    Pathology_masked = sitk.GetImageFromArray(np.uint8(Pathology_masked.astype(np.float)/np.max(Pathology_masked)*255))
    Pathology_masked.CopyInformation(mr_masked)
    sitk.WriteImage(Pathology_masked, './Pathology_masked.nii.gz')
    p_img = sitk.GetImageFromArray(np.uint8(Pathology_label_data.astype(np.float)/np.max(Pathology_label_data)*255))
    p_img.CopyInformation(img)
    sitk.WriteImage(p_img, './Pathology_label.nii.gz')

def postprocess(warped0='./warped_Pathology_ori_0.nii.gz',warped1='./warped_Pathology_ori_1.nii.gz',warped2='./warped_Pathology_ori_2.nii.gz'):
    warped0 = sitk.GetArrayFromImage(sitk.ReadImage(warped0))
    warped1 = sitk.GetArrayFromImage(sitk.ReadImage(warped1))
    warped2 = sitk.GetArrayFromImage(sitk.ReadImage(warped2))
    warped0 = np.expand_dims(warped0, axis=-1)
    warped1 = np.expand_dims(warped1, axis=-1)
    warped2 = np.expand_dims(warped2, axis=-1)
    a = np.concatenate([warped0,warped1,warped2], axis=-1)
    plt.imsave('./warepd_pa.png', a)
    
    return a

def command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print(f"\tLevel: {method.GetCurrentLevel()}")
        print(f"\tScales: {method.GetOptimizerScales()}")
    print(f"#{method.GetOptimizerIteration()}")
    print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
    print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")
    if (method.GetOptimizerConvergenceValue() != sys.float_info.max):
        print(f"\tConvergence Value: {method.GetOptimizerConvergenceValue():.5e}")

def command_multiresolution_iteration(method):
    print(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
    print("============= Resolution Change =============")


def registration(MOVING='./Pathology_masked.nii.gz', FIXED='./MR_masked.nii.gz', Transform='./trans.hdf'):
        
    fixed = sitk.ReadImage(FIXED, sitk.sitkFloat32)

    moving = sitk.ReadImage(MOVING, sitk.sitkFloat32)
    
    initialTx = sitk.CenteredTransformInitializer(fixed, moving,
                                                  sitk.AffineTransform(
                                                      fixed.GetDimension()))
    
    R = sitk.ImageRegistrationMethod()
    
    R.SetShrinkFactorsPerLevel([3, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 1])
    
    R.SetMetricAsJointHistogramMutualInformation(20)
    R.MetricUseFixedImageGradientFilterOff()
    
    R.SetOptimizerAsGradientDescent(learningRate=1.0,
                                    numberOfIterations=100,
                                    estimateLearningRate=R.EachIteration)
    R.SetOptimizerScalesFromPhysicalShift()
    
    R.SetInitialTransform(initialTx)
    
    R.SetInterpolator(sitk.sitkLinear)
    
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                 lambda: command_multiresolution_iteration(R))
    
    outTx1 = R.Execute(fixed, moving)
    
    print("-------")
    print(outTx1)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    
    displacementField = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
    displacementField.CopyInformation(fixed)
    displacementTx = sitk.DisplacementFieldTransform(displacementField)
    del displacementField
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0,
                                                varianceForTotalField=1.5)
    
    R.SetMovingInitialTransform(outTx1)
    R.SetInitialTransform(displacementTx, inPlace=True)
    
    R.SetMetricAsANTSNeighborhoodCorrelation(4)
    R.MetricUseFixedImageGradientFilterOff()
    
    R.SetShrinkFactorsPerLevel([3, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 1])
    
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetOptimizerAsGradientDescent(learningRate=1,
                                    numberOfIterations=300,
                                    estimateLearningRate=R.EachIteration)
    
    R.Execute(fixed, moving)
    
    print("-------")
    print(displacementTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    
    compositeTx = sitk.CompositeTransform([outTx1, displacementTx])
    sitk.WriteTransform(compositeTx, Transform)
    
    '''
    if ("SITK_NOSHOW" not in os.environ):
        sitk.Show(displacementTx.GetDisplacementField(), "Displacement Field")
    
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(compositeTx)
    
        out = resampler.Execute(moving)
        simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
        simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
        sitk.Show(cimg, "ImageRegistration1 Composition")
    '''

if __name__ == '__main__':
    if len(sys.argv) < 2:
        registration()    
    elif sys.argv[1] == 'pre':
        preprocess()
    elif sys.argv[1] == 'post':
        postprocess()
    
    