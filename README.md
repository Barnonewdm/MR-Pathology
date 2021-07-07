# MR-Pathology

This code is used to register MR and pathology images based on the SimpleITK package.

## Usage

* Run the registration, please: `python ./src/registration.py`
* Also, ANTs [scirpt](https://github.com/Barnonewdm/MR-Pathology/blob/main/src/ANTs.sh) is supplied. Please install the [ANTs](https://github.com/ANTsX/ANTs) before using the script.

## Results

| ID        | Dice (%)          | Hausdorff Distance  |
| ------------- |:-------------:| -----:|
| aaa0043D (itk-bspline) | 98.26      | 10.05 |
| aaa0043D (itk-bspline, Mattes) | 97.37      | 7.62 |
| aaa0043D (ANTs)        | 98.79      |  3.61 |
| aaa0051E (itk-bspline) | 91.52      | 30.61 |
| aaa0051E (itk-bspline, Mattes) | 96.92      | 5.00 |
| aaa0051E (ANTs)        | 96.04      | 30.87 |
