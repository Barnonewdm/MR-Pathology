# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 13:19:07 2021
@author: Dongming Wei
"""
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import cv2
import sys
from math import pi


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

def pad_and_resize(img, padding=(30,10), output_dims=(256,256), interpolation='linear'):
    
    if interpolation == 'linear':
        img = cv2.resize(img, np.subtract(output_dims, padding), interpolation=cv2.INTER_LINEAR)
    elif interpolation == 'nearest':
        img = cv2.resize(img, np.subtract(output_dims, padding), interpolation=cv2.INTER_NEAREST)
        
    img = padding_zeros_2D(img, padding)
        
    return img
    

def preprocess(MR_dir='../MR/aaa0043/09-16-2000-PELVISPROSTATE-50407/4.000000-T2AXIALSMFOV-51544', 
               Pathology_dir='../Pathology/aaa0043D', Results_Folder='./',
               output_dims=(256,256), mr_slice_id=12, padding=(30,10)):
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
    
    mr_image_data_slice = mr_img_data[mr_slice_id,::-1,]
    mr_label_img_data_slice = mr_label_img_data[mr_slice_id,:,:]
    
    mr_masked = mr_image_data_slice * mr_label_img_data_slice
    index = np.where(mr_label_img_data_slice == 1)
    
    mr_masked = np.uint8(mr_masked.astype(np.float)/np.max(mr_masked)*255)
    mr_masked = mr_masked[index[0].min()-5 : index[0].max()+10, index[1].min()-5 : index[1].max()+10]
    mr_masked = pad_and_resize(mr_masked,padding=padding)
    mr_masked = sitk.GetImageFromArray(mr_masked)
    sitk.WriteImage(mr_masked, Results_Folder + './MR_masked.nii.gz')
    
    img = np.uint8(mr_label_img_data_slice.astype(np.float)/np.max(mr_label_img_data_slice)*255)
    img = img[index[0].min()-5 : index[0].max()+10, index[1].min()-5 : index[1].max()+10]
    img = pad_and_resize(img, padding=padding)
    img = sitk.GetImageFromArray(img)
    img.CopyInformation(mr_masked)
    sitk.WriteImage(img, Results_Folder + './MR_label.nii.gz')
    
    
    # read Pathology image
    Pathology = Pathology_dir + '/img.png'
    Pathology_label = Pathology_dir + '/label.png'
    
    Pathology_data = pad_and_resize(rgb2gray(plt.imread(Pathology)), output_dims=output_dims, padding=padding, interpolation='linear')
    Pathology_label_data = pad_and_resize(rgb2gray(plt.imread(Pathology_label)), output_dims=output_dims, padding=padding, interpolation='nearest')
    Pathology_label_data[Pathology_label_data < Pathology_label_data.max()] = 0
    Pathology_label_data[Pathology_label_data > 0] = 1
    
    Pathology_masked = Pathology_data * Pathology_label_data
    #Pathology_masked = pad_and_resize(Pathology_masked, output_dims=output_dims)

    #Pathology_label_data = pad_and_resize(Pathology_label_data, output_dims=output_dims, interpolation='nearest')
    
    Pathology_ori = pad_and_resize(plt.imread(Pathology), padding=padding)
    for i in range(np.shape(Pathology_ori)[2]):
        P_ori_img = sitk.GetImageFromArray(Pathology_ori[:,:,i] * Pathology_label_data)
        sitk.WriteImage(P_ori_img, Results_Folder + './Pathology_ori_' + str(i) + '.nii.gz')
    
    Pathology_masked = sitk.GetImageFromArray(np.uint8(Pathology_masked.astype(np.float)/np.max(Pathology_masked)*255))
    Pathology_masked.CopyInformation(mr_masked)
    sitk.WriteImage(Pathology_masked, Results_Folder + './Pathology_masked.nii.gz')
    p_img = sitk.GetImageFromArray(np.uint8(Pathology_label_data.astype(np.float)/np.max(Pathology_label_data)*255))
    p_img.CopyInformation(img)
    sitk.WriteImage(p_img, Results_Folder + './Pathology_label.nii.gz')

def postprocess(Dir='./'):
    warped0= Dir + '/warped_pa_0.nii.gz'
    warped1= Dir + '/warped_pa_1.nii.gz'
    warped2= Dir + '/warped_pa_2.nii.gz'
    warped0 = sitk.GetArrayFromImage(sitk.ReadImage(warped0))
    warped1 = sitk.GetArrayFromImage(sitk.ReadImage(warped1))
    warped2 = sitk.GetArrayFromImage(sitk.ReadImage(warped2))
    warped0 = np.expand_dims(warped0, axis=-1)
    warped1 = np.expand_dims(warped1, axis=-1)
    warped2 = np.expand_dims(warped2, axis=-1)
    a = np.concatenate([warped0,warped1,warped2], axis=-1)
    plt.imsave(Dir + '/warepd_pa.png', a)
    
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
        
        R.SetMetricAsJointHistogramMutualInformation(5)#20
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
        
        R.SetMetricAsANTSNeighborhoodCorrelation(1)#4
        R.MetricUseFixedImageGradientFilterOff()
        
        R.SetShrinkFactorsPerLevel([3, 2, 1])
        R.SetSmoothingSigmasPerLevel([1, 1, 1])# 2,1,1
        
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetOptimizerAsGradientDescent(learningRate=1,
                                        numberOfIterations=500,
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
        resampler.SetDefaultPixelValue(0)
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
    def __init__(self, DIR='./'):
        self.DIR = DIR
    
    def command_iteration_affine(self, method):
        if (method.GetOptimizerIteration() == 0):
            print(f"\tLevel: {method.GetCurrentLevel()}")
            print(f"\tScales: {method.GetOptimizerScales()}")
        print(f"#{method.GetOptimizerIteration()}")
        print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
        print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")
        if (method.GetOptimizerConvergenceValue() != sys.float_info.max):
            print(f"\tConvergence Value: {method.GetOptimizerConvergenceValue():.5e}")
    
    def command_iteration(self, method, bspline_transform):
        if method.GetOptimizerIteration() == 0:
            # The BSpline is resized before the first optimizer
            # iteration is completed per level. Print the transform object
            # to show the adapted BSpline transform.
            print(bspline_transform)
    
        print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")
    
    def command_multiresolution_iteration(self, method):
        print(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
        print("============= Resolution Change =============")
    
    
    def command_multi_iteration(self, R):
        # The sitkMultiResolutionIterationEvent occurs before the
        # resolution of the transform. This event is used here to print
        # the status of the optimizer from the previous registration level.
        if R.GetCurrentLevel() > 0:
            print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
            print(f" Iteration: {R.GetOptimizerIteration()}")
            print(f" Metric value: {R.GetMetricValue()}")
    
        print("--------- Resolution Changing ---------")
    
    def affine(self, MOVING, FIXED, ind='affine'):
        fixed = sitk.ReadImage(FIXED, sitk.sitkFloat32)
    
        moving = sitk.ReadImage(MOVING, sitk.sitkFloat32)
        
        
        if ind == "affine":
            initialTx = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(fixed.GetDimension()))
        else:
            initialTx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        
        R = sitk.ImageRegistrationMethod()
        
        R.SetShrinkFactorsPerLevel([3, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 1])#2,1,1
        
        R.SetMetricAsJointHistogramMutualInformation()#20
        R.MetricUseFixedImageGradientFilterOff()
        
        R.SetOptimizerAsGradientDescent(learningRate=1.0,
                                        numberOfIterations=100,
                                        estimateLearningRate=R.EachIteration)
        R.SetOptimizerScalesFromPhysicalShift()
        
        R.SetInitialTransform(initialTx)
        
        R.SetInterpolator(sitk.sitkLinear)
        
        R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration_affine(R))
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                     lambda: self.command_multiresolution_iteration(R))
        
        outTx1 = R.Execute(fixed, moving)
        
        return outTx1
        
    def reg(self, MOVING='./Pathology_masked.nii.gz', FIXED='./MR_masked.nii.gz', Transform='./trans.hdf'):
    
        fixed = sitk.ReadImage(FIXED, sitk.sitkFloat32)
        
        moving = sitk.ReadImage(MOVING, sitk.sitkFloat32)
        
        transformDomainMeshSize = [1] * fixed.GetDimension()
        tx = sitk.BSplineTransformInitializer(fixed,
                                              transformDomainMeshSize)
        
        print(f"Initial Number of Parameters: {tx.GetNumberOfParameters()}")
        
        
        R = sitk.ImageRegistrationMethod()
        
        R.SetMetricAsJointHistogramMutualInformation()
        
        R.SetOptimizerAsGradientDescentLineSearch(5.0, #5.0
                                                  100, #100
                                                  convergenceMinimumValue=1e-4,
                                                  convergenceWindowSize=5)#5
        
        R.SetInterpolator(sitk.sitkLinear)
        
        # affine transform
        affine = self.affine(MOVING, FIXED, ind='rotate')
        
        
        R.SetMovingInitialTransform(affine)
        
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
        
        compositeTx = sitk.CompositeTransform([affine, outTx])
        sitk.WriteTransform(compositeTx, Transform)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(compositeTx)
    
        out = resampler.Execute(moving)
        sitk.WriteImage(out, self.DIR + '/itk_bspline_out.nii.gz')
        
        warper(MOVING= self.DIR + './Pathology_label.nii.gz', FIXED= self.DIR + './MR_label.nii.gz', SAVING = self.DIR + '/warped_pathology_label.nii.gz', Transform= Transform)


def warper(MOVING, FIXED, Transform, SAVING = 'warped_label.nii.gz', Interpolation = sitk.sitkNearestNeighbor):
    moving = sitk.ReadImage(MOVING, sitk.sitkFloat32)
    fixed = sitk.ReadImage(FIXED, sitk.sitkFloat32)
    transform = sitk.ReadTransform(Transform)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(Interpolation)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    out = resampler.Execute(moving)
    sitk.WriteImage(out, SAVING)
    
def dice(FIXED='./MR_label.nii.gz', MOVED='./warped_label.nii.gz',label=255):
    label=int(label)
    if label==0:
        fixed = sitk.ReadImage(FIXED)
        moved = sitk.ReadImage(MOVED)
        
        fix_data = sitk.GetArrayFromImage(fixed)
        moved_data = sitk.GetArrayFromImage(moved)
        MO = np.sum(moved_data[(fix_data>label) * (moved_data>label)])*2.0/(np.sum(fix_data[fix_data>label])
                + np.sum(moved_data[moved_data>label]))
        print("Dice: %.2f" % (100*MO) + "%")
        return MO
    else:
        label=int(label)
        fixed = sitk.ReadImage(FIXED)
        moved = sitk.ReadImage(MOVED)
        
        fix_data = sitk.GetArrayFromImage(fixed)
        moved_data = sitk.GetArrayFromImage(moved)
        if np.sum(fix_data[fix_data==label]) + np.sum(moved_data[moved_data==label]) ==0:
            print("[Warning]: No this label!")
        else:
            MO = np.sum(moved_data[(fix_data==label) * (moved_data==label)])*2.0/(np.sum(fix_data[fix_data==label])
                    + np.sum(moved_data[moved_data==label]))
            print("Dice: %.2f" % (100*MO) + "%")
            return MO
        
def Hausdorff_distance(FIXED='./MR_label.nii.gz', MOVED='./warped_label.nii.gz'):
    
    fix = sitk.ReadImage(FIXED, sitk.sitkFloat64)
    moved = sitk.ReadImage(MOVED, sitk.sitkFloat64)
    
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    # Use reference1, larger inner annulus radius, the surface based computation
    # has a smaller difference. 
    hausdorff_distance_filter.Execute(fix, moved)
    print('HausdorffDistanceImageFilter result (reference1-segmentation): %.2f ' % 
          hausdorff_distance_filter.GetHausdorffDistance())
            
if __name__ == '__main__':
    Saving_Dir='../Results/aa0051'
    if len(sys.argv) < 2:
        ##########################
        # We have several options for the registration model
        # 1. displacement_registration
        # 2. demons_registration
        # 3. bspline_registration
        ##########################
        preprocess(MR_dir='../MR/aaa0051/07-02-2000-PELVISPROSTATE-97855/4.000000-T2 AXIAL SM FOV-36207', Pathology_dir='../Pathology/aaa0051E', Results_Folder=Saving_Dir, padding=(32,16))
        #preprocess(MR_dir='../MR/aaa0043/09-16-2000-PELVISPROSTATE-50407/4.000000-T2AXIALSMFOV-51544', Pathology_dir='../Pathology/aaa0043D', Results_Folder=Saving_Dir, padding=(6,6))
        reg = bspline_registration(DIR=Saving_Dir)
        reg.reg(MOVING= Saving_Dir + './Pathology_masked.nii.gz', FIXED= Saving_Dir + './MR_masked.nii.gz', Transform= Saving_Dir + './trans.hdf')
        dice(FIXED= Saving_Dir + '/MR_label.nii.gz', MOVED=Saving_Dir + '/warped_pathology_label.nii.gz')
        Hausdorff_distance(FIXED= Saving_Dir + '/MR_label.nii.gz', MOVED= Saving_Dir + '/warped_pathology_label.nii.gz')
        #postprocess('../Results/aa0051/ANTs')
    
    elif sys.argv[1] == 'pre':
        preprocess(MR_dir='../MR/aaa0051/07-02-2000-PELVISPROSTATE-97855/4.000000-T2 AXIAL SM FOV-36207',
                   Pathology_dir='../Pathology/aaa0051E')
    elif sys.argv[1] == 'post':
        postprocess()
