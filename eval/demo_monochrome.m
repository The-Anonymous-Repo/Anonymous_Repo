clear;

%% settings
addpath(genpath('Functions'));
addpath(genpath('BM3D'));

% Result folder
if exist('Results_monochrome') == 0
    mkdir('Results_monochrome')
end

% Load Data (using green-channel images of our color-polarization dataset)
% Use 'GT_0','GT_45','GT_90','GT_135' as ground-truth intensity images
% Use 'Test_0','Test_45','Test_90','Test_135' to create input mosaic image
data = sprintf('InputPDD/ImageMat/High/Scene1.mat');
load(data);
                        
GT_90 = GT_90(:,:,2);  
GT_45 = GT_45(:,:,2);        
GT_135 = GT_135(:,:,2);        
GT_0 = GT_0(:,:,2);
          
Test_90 = Test_90(:,:,2);        
Test_45 = Test_45(:,:,2);        
Test_135 = Test_135(:,:,2);        
Test_0 = Test_0(:,:,2);
     
% Image size
s1 = 768;      
s2 = 1024; 
        
%% Make mosaic mask       
% Mask for 2x2 regular pattern        
mask = [];        
for i = 1:2            
    for j = 1:2                        
        temp_mask = zeros(s1,s2);                       
        temp_mask(i:2:end,j:2:end) = 1;                       
        mask = cat(3,mask,temp_mask);                
    end        
end
        
mask_P90 = mask(:,:,1);       
mask_P45 = mask(:,:,2);       
mask_P135 = mask(:,:,3);     
mask_P0 = mask(:,:,4);
        
%% Calculate Stokes parameters(original)        
A = [1,1,0,0;    
    1,0,1,0;    
    1,-1,0,0;    
    1,0,-1,0];

[S0, S1, S2, DoP, AoP] = Process_images_stokes(GT_0,GT_45,GT_90,GT_135,A);

% Save original images
imwrite(GT_90,sprintf('Results_monochrome/GT_90.png'));
imwrite(GT_45,sprintf('Results_monochrome/GT_45.png'));
imwrite(GT_135,sprintf('Results_monochrome/GT_135.png'));
imwrite(GT_0,sprintf('Results_monochrome/GT_0.png'));
imwrite(S0,sprintf('Results_monochrome/GT_S0.png'));
imwrite(aolp_dolp(AoP,sqrt(DoP)),sprintf('Results_monochrome/GT_AoP_DoP.png'));
                     
%% Polarization denoising and demosaicking   
% Make polarization mosaic image      
MPFA = zeros(s1,s2);      
MPFA(1:2:end,1:2:end,:) = Test_90(1:2:end,1:2:end);      
MPFA(1:2:end,2:2:end,:) = Test_45(1:2:end,2:2:end);      
MPFA(2:2:end,1:2:end,:) = Test_135(2:2:end,1:2:end);        
MPFA(2:2:end,2:2:end,:) = Test_0(2:2:end,2:2:end);

% Denoising by PFCD            
noiselevel = 7.31; % High noise-level condition

pattern = '1';        
[MPFA_denoised, P] = MPFAdenoise( MPFA, noiselevel, pattern );

% Demosaicking by IGRI-2
[Dem_0, Dem_45, Dem_90, Dem_135] = IGRI2(MPFA_denoised,eps,mask_P0,mask_P45,mask_P90,mask_P135);
        
%% Calculate stokes parameters
[Dem_S0,Dem_S1,Dem_S2,Dem_DoP,Dem_AoP] = Process_images_stokes(Dem_0,Dem_45,Dem_90,Dem_135,A);

% Save demosaicked images 
imwrite(Dem_90,sprintf('Results_monochrome/Dem_90.png'));
imwrite(Dem_45,sprintf('Results_monochrome/Dem_45.png'));
imwrite(Dem_135,sprintf('Results_monochrome/Dem_135.png'));
imwrite(Dem_0,sprintf('Results_monochrome/Dem_0.png'));
imwrite(Dem_S0,sprintf('Results_monochrome/Dem_S0.png'));
imwrite(aolp_dolp(Dem_AoP,sqrt(Dem_DoP)),sprintf('Results_monochrome/Dem_AoP_DoP.png'));

%% Calculate PSNR and RMSE of angle error     
psnr_90 = impsnr(GT_90, Dem_90, 1, 15);      
psnr_45 = impsnr(GT_45, Dem_45, 1, 15);         
psnr_135 = impsnr(GT_135, Dem_135, 1, 15);        
psnr_0 = impsnr(GT_0, Dem_0, 1, 15);  

psnr_S0 = impsnr(S0, Dem_S0, 1, 15);        
psnr_S1 = impsnr(S1, Dem_S1, 1, 15);        
psnr_S2 = impsnr(S2, Dem_S2, 1, 15);       
psnr_DOLP = impsnr(DoP, Dem_DoP, 1, 15);       
angleerror = angleerror_AOLP(AoP, Dem_AoP, 15);
       
result = [psnr_0,psnr_45,psnr_90,psnr_135,psnr_S0,psnr_S1,psnr_S2,psnr_DOLP,angleerror]    
csvwrite('Results_monochrome/monochrome.csv',result);