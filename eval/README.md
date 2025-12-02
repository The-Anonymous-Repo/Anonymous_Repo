# Polarization Denoising and Demosaicking Codes
This is the source code of our proposed interpolation-based monochrome and color polarization denoising and demosaicking method, PFCD + IGRI-2, published in ICIP2025. <a href="http://www.ok.sc.e.titech.ac.jp/res/PolarDem/PDD.html" target="_blank">[Project]</a>.
 
## Demo

### - Settings
This denoising algorithm requires a gray-scale denoiser, and we used BM3D for denoising.
Please download the BM3D MATLAB code <a href="https://webpages.tuni.fi/foi/GCF-BM3D/BM3D.zip">[Link]</a>.
Or, you can just run sample_download.m to obtain the codes. 

### - Dataset
Our 24-channel noisy-color-polarization dataset is downloadable from our project page <a href="http://www.ok.sc.e.titech.ac.jp/res/PolarDem/PDD.html" target="_blank">[Project]</a>.
Our 24-channel noisy-color-polarization dataset contains 12-channel noise-free color polarization data and 12-channel noisy color polarization data.

In such a structure, 
Scene(number) - GT_0
              - GT_45
              - GT_90
              - GT_135
              - Test_0
              - Test_45
              - Test_90
              - Test_135

12-channel noise-free color-polarization data:  GT_0, GT_45, GT_90, GT_135. Use these data as ground-truth intensity images.<br>
12-channel noisy color-polarization data: Test_0, Test_45, Test_90, Test_135. Use these data to make input polarization mosaic images.

Here, we provide a 24-channel full noisy-color-polarization dataset in three different noise-level conditions, which are low, medium, and high.
The average noise level for each condition can be referred to Table 1 in our paper. 
To obtain the same results as our main paper and supplementary material, 
please use the correct dataset (low, medium, or high) with the corresponding noise level parameter.

### - sample_download.m
To run our demo codes, please first run the sample_download.m code to obtain the sample 24-channel noisy-color-polarization data and BM3D settings to be used for the demo.


### - demo_monochrome.m
#### Input
8-channel monochrome-polarization data (G channel data): GT_0, GT_45, GT_90, GT_135, Test_0, Test_45, Test_90, Test_135

#### Mosaic pattern
90 &nbsp; 45<br>
135  &nbsp; 0<br>

### Denoising 
Set the noise-level parameter to be used to control denoising power.<br>
For MPFA denoising, noise-level for (G channel data) being used: sigma_g<br>
In the code, we provide the noise-level parameter for high noise-level conditions.

#### Output
Demosaicked-denoised monochrome images for each polarization direction: Dem_0, Dem_45, Dem_90, Dem_135<br>
Stokes parameter images derived from the demosaicked-denoised images: Dem_S0, Dem_S1, Dem_S2, Dem_DoP, Dem_AoP<br>
CSV file containing PSNR values and angle RMSE for Table 2 in the paper

### - demo_color.m
#### Input
24-channel noisy-color-polarization mat data: mat = cat(3, GT_0, GT_45, GT_90, GT_135, Test_0, Test_45, Test_90, Test_135)

#### Mosaic pattern
R_90 &nbsp;&nbsp; R_45 &nbsp; G_90 &nbsp; G_45<br>
R_135 &nbsp; R_0 &nbsp; G_135 &nbsp; G_0<br>
G_90 &nbsp;&nbsp; G_45 &nbsp; B_90 &nbsp; B_45<br>
G_135 &nbsp; G_0 &nbsp; B_135 &nbsp; B_0<br>

### Denoising 
Set the noise-level parameter to be used to control denoising power.<br>
For CFA denoising, noise-level for (RGB channel data) being used: [sigma_r,sigma_g,sigma_b]<br>
In the code, we provide the noise-level parameter for high noise-level conditions.


#### Output
Demosaicked-denoised RGB images for each polarization direction: Dem_0, Dem_45, Dem_90, Dem_135<br>
Stokes parameter images derived from the demosaicked-denoised RGB images: Dem_S0, Dem_S1, Dem_S2, Dem_DoP, Dem_AoP<br>
CSV file containing PSNR values and angle RMSE for Table 3 in the paper


## Reference
The code is available for research purposes only. If you use this code for publications, please cite the following papers.

"Polarization Denoising and Demosaicking: Dataset and Baseline Method"<br>
Muhamad Daniel Ariff Bin Abdul Rahman, Yusuke Monno, Masayuki Tanaka, and Masatoshi Okutomi<br>
IEEE International Conference on Image Processing (ICIP), pp.2524-2529, September, 2025.
