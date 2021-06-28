# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:43:23 2021

@author: shaon
"""

import SimpleITK as sitk
import sys
import os
from math import pi


def command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print("Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")


if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "<fixedImageFilter> <movingImageFile>",
          "<outputTransformFile>")
    sys.exit(1)

fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)

moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)

R = sitk.ImageRegistrationMethod()

R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

sample_per_axis = 12
if fixed.GetDimension() == 2:
    tx = sitk.Euler2DTransform()
    # Set the number of samples (radius) in each dimension, with a
    # default step size of 1.0
    R.SetOptimizerAsExhaustive([sample_per_axis // 2, 0, 0])
    # Utilize the scale to set the step size for each dimension
    R.SetOptimizerScales([2.0 * pi / sample_per_axis, 1.0, 1.0])
elif fixed.GetDimension() == 3:
    tx = sitk.Euler3DTransform()
    R.SetOptimizerAsExhaustive([sample_per_axis // 2, sample_per_axis // 2,
                                sample_per_axis // 4, 0, 0, 0])
    R.SetOptimizerScales(
        [2.0 * pi / sample_per_axis, 2.0 * pi / sample_per_axis,
         2.0 * pi / sample_per_axis, 1.0, 1.0, 1.0])

# Initialize the transform with a translation and the center of
# rotation from the moments of intensity.
tx = sitk.CenteredTransformInitializer(fixed, moving, tx)

R.SetInitialTransform(tx)

R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

outTx = R.Execute(fixed, moving)

print("-------")
print(outTx)

print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
print(f" Iteration: {R.GetOptimizerIteration()}")
print(f" Metric value: {R.GetMetricValue()}")

sitk.WriteTransform(outTx, sys.argv[3])

if ("SITK_NOSHOW" not in os.environ):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    sitk.Show(cimg, "ImageRegistrationExhaustive Composition")