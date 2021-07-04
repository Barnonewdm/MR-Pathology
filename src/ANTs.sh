fixed=./MR_masked.nii.gz
moving=./Pathology_masked.nii.gz

/home/data/weidongming/ANTs-2.1.0-Linux/bin/antsRegistration --dimensionality 2 --float 0 --output [ANTs_warped_img,ANTs_warped.nii.gz] --interpolation Linear --winsorize-image-intensities [0.005,0.995] --use-historgram-matching 0 --initial-moving-transform [$fixed,$moving,1] --transform Rigid[0.1] --metric MI[$fixed,$moving,1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform Affine[0.1] --metric MI[$fixed,$moving,1,32,Regular,0.25] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform SyN[0.1,3,0] --metric MI[$fixed,$moving,1,32,Regular,0.25] --convergence [100x70x50x20,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox

/home/data/weidongming/ANTs-2.1.0-Linux/bin/WarpImageMultiTransform 2 Pathology_label.nii.gz warped_pathology_label.nii.gz -R $fixed ANTs_warped_img1Warp.nii.gz ANTs_warped_img0GenericAffine.mat

for i in {0..2}; do echo $i;  /home/data/weidongming/ANTs-2.1.0-Linux/bin/WarpImageMultiTransform 2 Pathology_ori_${i}.nii.gz warped_pa_${i}.nii.gz -R $fixed ANTs_warped_img1Warp.nii.gz ANTs_warped_img0GenericAffine.mat ; done
