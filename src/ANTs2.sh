
antsRegistration=/home/winter/Tools/ANTs-1.9.v4-Linux/bin/antsRegistration
WarpImageMultiTransform=/home/winter/Tools/ANTs-1.9.v4-Linux/bin/WarpImageMultiTransform
save_name=moving_to_

python registration.py pre

moving=./Pathology_masked.nii.gz
fixed=./MR_masked.nii.gz
label=./Pathology_ori

while getopts ":m:t:l:" opt; do
	case ${opt} in
		m )
			moving=$OPTARG
			;;
		f )
			fixed=$OPTARG
			;;
		l )
			label=$OPTARG
			;;
	esac
done

echo "[Registration Started]"
echo "[Moving image: ${moving}]"
echo "[Fixed image: ${fixed}]"

$antsRegistration --dimensionality 2 --float 0 \
        --output [${save_name},${save_name}Warped.nii.gz] \
        --interpolation Linear \
        --winsorize-image-intensities [0.005,0.995] \
        --use-histogram-matching 0 \
        --initial-moving-transform [$fixed,$moving,1] \
        --transform Rigid[0.1] \
        --metric MI[$fixed,$moving,1,32,Regular,0.25] \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        --transform Affine[0.1] \
        --metric MI[$fixed,$moving,1,32,Regular,0.25] \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        --transform SyN[0.1,3,0] \
        --metric MI[$fixed,$moving,1,32,Regular,0.25] \
        --convergence [100x70x50x20,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox
for i in {0..2}
do
	$WarpImageMultiTransform 2 ${label}_${i}.nii.gz warped_${label}_${i}.nii.gz -R $fixed ${save_name}1Warp.nii.gz ${save_name}0GenericAffine.mat 
done

python registration.py post

echo "[Registration Done]"

