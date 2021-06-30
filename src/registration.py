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

def pad_and_resize(img, padding=(600,600), output_dims=(256,256), interpolation='linear'):
    img = padding_zeros_2D(img, padding)
    if interpolation == 'linear':
        img = cv2.resize(img, output_dims, interpolation=cv2.INTER_LINEAR)
    elif interpolation == 'nearest':
        img = cv2.resize(img, output_dims, interpolation=cv2.INTER_NEAREST)
        
    return img
    

def preprocess(MR_dir='../MR/aaa0043/09-16-2000-PELVISPROSTATE-50407/4.000000-T2AXIALSMFOV-51544', 
               Pathology_dir='../Pathology/aaa0043D', Results_Folder='./',
               output_dims=(256,256)):
    ###########
    # function: the preprocess function read the original MR and pathology images, and perform image masking,
    # cropping, padding, resizing, and intensity normalization
    # usage: preprocess(MR_dir='./MR', Pathology_dir='./Pathology', output_dims=(256,256))
    # output: 
    ###########
    # read MR image slice
    MR = MR_dir + '/T2.nii'
    MR_LABEL = MR_dir + '/T2_label.nii'
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
    plt.imsave(Results_Folder + './MR_masked.png', mr_masked, cmap='gray')
    mr_masked = sitk.GetImageFromArray(mr_masked)
    sitk.WriteImage(mr_masked, Results_Folder + './MR_masked.nii.gz')
    img = np.uint8(mr_label_img_data_slice.astype(np.float)/np.max(mr_label_img_data_slice)*255)
    img = img[index[0].min()-5 : index[0].max()+5, index[1].min()-5 : index[1].max()+5]
    
    img = cv2.resize(img, output_dims, interpolation=cv2.INTER_NEAREST)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, Results_Folder + './MR_label.nii.gz')
    
    
    # read Pathology image
    Pathology = Pathology_dir + '/img.png'
    Pathology_label = Pathology_dir + '/label.png'
    Pathology_ori = plt.imread(Pathology)
    #Pathology_ori = padding_zeros_2D(Pathology_ori, (600, 600))
    #Pathology_ori = cv2.resize(Pathology_ori, output_dims, interpolation=cv2.INTER_LINEAR)
    Pathology_ori = pad_and_resize(Pathology_ori)
    Pathology_data = rgb2gray(plt.imread(Pathology))
    Pathology_label_data = rgb2gray(plt.imread(Pathology_label))
    Pathology_label_data[Pathology_label_data < Pathology_label_data.max()] = 0
    Pathology_label_data[Pathology_label_data >0] = 1
    
    Pathology_masked = Pathology_data * Pathology_label_data
    Pathology_masked = pad_and_resize(Pathology_masked, output_dims=output_dims)

    Pathology_label_data = pad_and_resize(Pathology_label_data, output_dims=output_dims, interpolation='nearest')
    
    for i in range(np.shape(Pathology_ori)[2]):
        P_ori_img = sitk.GetImageFromArray(Pathology_ori[:,:,i] * Pathology_label_data)
        #P_ori_img.CopyInformation(img)
        sitk.WriteImage(P_ori_img, Results_Folder + './Pathology_ori_' + str(i) + '.nii.gz')
    
    plt.imsave(Results_Folder + './Pathology_masked.png', Pathology_masked, cmap='gray')
    Pathology_masked = sitk.GetImageFromArray(np.uint8(Pathology_masked.astype(np.float)/np.max(Pathology_masked)*255))
    Pathology_masked.CopyInformation(mr_masked)
    sitk.WriteImage(Pathology_masked, Results_Folder + './Pathology_masked.nii.gz')
    p_img = sitk.GetImageFromArray(np.uint8(Pathology_label_data.astype(np.float)/np.max(Pathology_label_data)*255))
    p_img.CopyInformation(img)
    sitk.WriteImage(p_img, Results_Folder + './Pathology_label.nii.gz')

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

class displacement_registration():
    def command_iteration(self, method):
        if (method.GetOptimizerIteration() == 0):
            print(f"\tLevel: {method.GetCurrentLevel()}")
            print(f"\tScales: {method.GetOptimizerScales()}")
        print(f"#{method.GetOptimizerIteration()}")
        print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
        print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")
        if (method.GetOptimizerConvergenceValue() != sys.float_info.max):
            print(f"\tConvergence Value: {method.GetOptimizerConvergenceValue():.5e}")
    
    def command_multiresolution_iteration(self, method):
        print(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
        print("============= Resolution Change =============")
    
    
    def reg(self, MOVING='./Pathology_masked.nii.gz', FIXED='./MR_masked.nii.gz', Transform='./trans.hdf'):
            
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
        
        R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(R))
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                     lambda: self.command_multiresolution_iteration(R))
        
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
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(compositeTx)
        
        out = resampler.Execute(moving)
        sitk.WriteImage(out, 'itk_displacement_out.nii.gz')

class demons_registration():
    def command_iteration(self, filter):
        print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    
    def reg(self, MOVING='./Pathology_masked.nii.gz', FIXED='./MR_masked.nii.gz', Transform='./demons_trans.hdf'):
    
        fixed = sitk.ReadImage(FIXED, sitk.sitkFloat32)
        
        moving = sitk.ReadImage(MOVING, sitk.sitkFloat32)
        
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        moving = matcher.Execute(moving, fixed)
        
        # The basic Demons Registration Filter
        # Note there is a whole family of Demons Registration algorithms included in
        # SimpleITK
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(50)
        # Standard deviation for Gaussian smoothing of displacement field
        demons.SetStandardDeviations(1.0)
        
        demons.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(demons))
        
        displacementField = demons.Execute(fixed, moving)
        
        print("-------")
        print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
        print(f" RMS: {demons.GetRMSChange()}")
        
        outTx = sitk.DisplacementFieldTransform(displacementField)
        
        sitk.WriteTransform(outTx, Transform)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)
        
        out = resampler.Execute(moving)
        sitk.WriteImage(out, 'itk_demons_out.nii.gz')


class bspline_registration():
    def command_iteration(self, method, bspline_transform):
        if method.GetOptimizerIteration() == 0:
            # The BSpline is resized before the first optimizer
            # iteration is completed per level. Print the transform object
            # to show the adapted BSpline transform.
            print(bspline_transform)
    
        print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")
    
    
    def command_multi_iteration(self, R):
        # The sitkMultiResolutionIterationEvent occurs before the
        # resolution of the transform. This event is used here to print
        # the status of the optimizer from the previous registration level.
        if R.GetCurrentLevel() > 0:
            print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
            print(f" Iteration: {R.GetOptimizerIteration()}")
            print(f" Metric value: {R.GetMetricValue()}")
    
        print("--------- Resolution Changing ---------")
        
    def reg(self, MOVING='./Pathology_masked.nii.gz', FIXED='./MR_masked.nii.gz', Transform='./trans.hdf'):
    
        fixed = sitk.ReadImage(FIXED, sitk.sitkFloat32)
        
        moving = sitk.ReadImage(MOVING, sitk.sitkFloat32)
        
        transformDomainMeshSize = [2] * fixed.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed,
                                              transformDomainMeshSize)
        
        print(f"Initial Number of Parameters: {tx.GetNumberOfParameters()}")
        
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsJointHistogramMutualInformation()
        
        R.SetOptimizerAsGradientDescentLineSearch(5.0,
                                                  100,
                                                  convergenceMinimumValue=1e-4,
                                                  convergenceWindowSize=5)
        
        R.SetInterpolator(sitk.sitkLinear)
        
        R.SetInitialTransformAsBSpline(tx,
                                       inPlace=True,
                                       scaleFactors=[1, 2, 5])
        R.SetShrinkFactorsPerLevel([4, 2, 1])
        R.SetSmoothingSigmasPerLevel([4, 2, 1])
        
        R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(R, tx))
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                     lambda: self.command_multi_iteration(R))
        
        outTx = R.Execute(fixed, moving)
        
        print("-------")
        print(tx)
        print(outTx)
        print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {R.GetOptimizerIteration()}")
        print(f" Metric value: {R.GetMetricValue()}")
        
        sitk.WriteTransform(outTx, Transform)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)
    
        out = resampler.Execute(moving)
        sitk.WriteImage(out, 'itk_bspline_out.nii.gz')
        
            
if __name__ == '__main__':
    if len(sys.argv) < 2:
        #preprocess(MR_dir='/home/winter/Downloads/MR-Pathology/MR/aaa0051/07-02-2000-PELVISPROSTATE-97855/4.000000-T2 AXIAL SM FOV-36207',
         #          Pathology_dir='/home/winter/Downloads/MR-Pathology/Pathology/aaa0051E')
        # We have several options for the registration model
        # 1. displacement_registration
        # 2. demons_registration
        # 3. bspline_registration
        reg = bspline_registration()
        reg.reg()
    elif sys.argv[1] == 'pre':
        preprocess()
    elif sys.argv[1] == 'post':
        postprocess()