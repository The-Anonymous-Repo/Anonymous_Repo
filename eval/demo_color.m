clear;

%% settings
addpath(genpath('Functions'));
addpath(genpath('BM3D'));

% Result folder
if exist('Results_color') == 0
    mkdir('Results_color')
end

% Load Data 
% Use 'GT_0','GT_45','GT_90','GT_135' as ground-truth intensity images
% Use 'Test_0','Test_45','Test_90','Test_135' to create input mosaic image
data = sprintf('InputPDD/ImageMat/High/Scene1.mat');
load(data);

% Image size
s1 = 768;      
s2 = 1024; 

%% make mosaic mask
% mask for 4x4 regular pattern      
mask = [];        
for i = 1:4            
    for j = 1:4                
        temp_mask = zeros(s1, s2, 1);                
        temp_mask(i:4:end, j:4:end, 1) = 1;                
        mask = cat(3, mask, temp_mask);            
    end        
end
        
% Mask for R channel       
mask_R90 = mask(:,:,9);      
mask_R45 = mask(:,:,10);       
mask_R135 = mask(:,:,13);      
mask_R0 = mask(:,:,14); 

% Mask for G channel
mask_G90 = mask(:,:,1) + mask(:,:,11);      
mask_G45 = mask(:,:,2) + mask(:,:,12);       
mask_G135 = mask(:,:,5) + mask(:,:,15);      
mask_G0 = mask(:,:,6) + mask(:,:,16);  

% Mask for B channel
mask_B90 = mask(:,:,3);       
mask_B45 = mask(:,:,4);      
mask_B135 = mask(:,:,7);       
mask_B0 = mask(:,:,8); 

% Mask for each polarization angle 
mask_P90 = mask_R90 + mask_G90 + mask_B90;       
mask_P45 = mask_R45 + mask_G45 + mask_B45;     
mask_P135 = mask_R135 + mask_G135 + mask_B135;       
mask_P0 = mask_R0 + mask_G0 + mask_B0;
               
%% Calculate Stokes Parameters (Ground Truth)        
A = [1,1,0,0;            
    1,0,1,0;            
    1,-1,0,0;            
    1,0,-1,0];
       
[R_S0, R_S1, R_S2, R_DoP, R_AoP] = Process_images_stokes(GT_0(:,:,1), GT_45(:,:,1), GT_90(:,:,1), GT_135(:,:,1), A);        
[G_S0, G_S1, G_S2, G_DoP, G_AoP] = Process_images_stokes(GT_0(:,:,2), GT_45(:,:,2), GT_90(:,:,2), GT_135(:,:,2), A);       
[B_S0, B_S1, B_S2, B_DoP, B_AoP] = Process_images_stokes(GT_0(:,:,3), GT_45(:,:,3), GT_90(:,:,3), GT_135(:,:,3), A);

% save images
imwrite(GT_90,sprintf('Results_color/GT_90.png'));
imwrite(GT_45,sprintf('Results_color/GT_45.png'));
imwrite(GT_135,sprintf('Results_color/GT_135.png'));
imwrite(GT_0,sprintf('Results_color/GT_0.png'));
S0 = cat(3,R_S0,G_S0,B_S0);
imwrite(S0,sprintf('Results_color/GT_S0.png'));

imwrite(R_S0,sprintf('Results_color/GT_R_S0.png'));
imwrite(aolp_dolp(R_AoP,sqrt(R_DoP)),sprintf('Results_color/GT_R_AoP_DoP.png'));

imwrite(G_S0,sprintf('Results_color/GT_G_S0.png'));
imwrite(aolp_dolp(G_AoP,sqrt(G_DoP)),sprintf('Results_color/GT_G_AoP_DoP.png'));

imwrite(B_S0,sprintf('Results_color/GT_B_S0.png'));
imwrite(aolp_dolp(B_AoP,sqrt(B_DoP)),sprintf('Results_color/GT_B_AoP_DoP.png'));

%% Denoising and Bayer demosaicking
% make PCFA image     
R_90 = mask_R90 .* Test_90(:,:,1);      
R_45 = mask_R45 .* Test_45(:,:,1);        
R_135 = mask_R135 .* Test_135(:,:,1);        
R_0 = mask_R0 .* Test_0(:,:,1);
              
G_90 = mask_G90 .* Test_90(:,:,2);       
G_45 = mask_G45 .* Test_45(:,:,2);        
G_135 = mask_G135 .* Test_135(:,:,2);        
G_0 = mask_G0 .* Test_0(:,:,2);
                
B_90 = mask_B90 .* Test_90(:,:,3);       
B_45 = mask_B45 .* Test_45(:,:,3);        
B_135 = mask_B135 .* Test_135(:,:,3);       
B_0 = mask_B0 .* Test_0(:,:,3);
              
PCFA = R_90 + R_45 + R_135 + R_0 + G_90 + G_45 + G_135 + G_0 + B_90 + B_45 + B_135 + B_0;
   
% make Bayer polarization images      
Bayer_90 = PCFA(1:2:end,1:2:end);      
Bayer_45 = PCFA(1:2:end,2:2:end);       
Bayer_135 = PCFA(2:2:end,1:2:end);        
Bayer_0 = PCFA(2:2:end,2:2:end);
        
% Denoising by PFCD    
noiselevel = [8.62, 7.31, 15.79]; % High noise-level condition

pattern = 'gbrg'; sigma = 1; eps = 1e-32;       
Bayer_90_denoise = CFAdenoise( Bayer_90, noiselevel, pattern );      
Bayer_45_denoise = CFAdenoise( Bayer_45, noiselevel, pattern );       
Bayer_135_denoise = CFAdenoise( Bayer_135, noiselevel, pattern );        
Bayer_0_denoise = CFAdenoise( Bayer_0, noiselevel, pattern );
      
% Bayer demosaicking by Residual Interpolation (RI)    
BayerDem_90 = demosaick(repmat(Bayer_90_denoise,[1,1,3]),pattern,sigma, eps);       
BayerDem_45 = demosaick(repmat(Bayer_45_denoise,[1,1,3]),pattern,sigma, eps);       
BayerDem_135 = demosaick(repmat(Bayer_135_denoise,[1,1,3]),pattern,sigma, eps);       
BayerDem_0 = demosaick(repmat(Bayer_0_denoise,[1,1,3]),pattern,sigma, eps); 
        
%% Polarization demosaicking   
% make RGB polarization mosaic image 
BayerDem_RGB = zeros(s1, s2, 3);      
BayerDem_RGB(1:2:end, 1:2:end, :) = BayerDem_90;      
BayerDem_RGB(1:2:end, 2:2:end, :) = BayerDem_45;        
BayerDem_RGB(2:2:end, 1:2:end, :) = BayerDem_135;        
BayerDem_RGB(2:2:end, 2:2:end, :) = BayerDem_0;

% Polarization demosaicking 
[Dem_0, Dem_45, Dem_90, Dem_135] = IGRI2(BayerDem_RGB, eps, mask_P0, mask_P45, mask_P90, mask_P135);
        
%% Stokes Parameters (Demosaicked)
[Dem_R_S0, Dem_R_S1, Dem_R_S2, Dem_R_DoP, Dem_R_AoP] ...
    = Process_images_stokes(Dem_0(:,:,1), Dem_45(:,:,1), Dem_90(:,:,1), Dem_135(:,:,1), A);
[Dem_G_S0, Dem_G_S1, Dem_G_S2, Dem_G_DoP, Dem_G_AoP] ...
    = Process_images_stokes(Dem_0(:,:,2), Dem_45(:,:,2), Dem_90(:,:,2), Dem_135(:,:,2), A);       
[Dem_B_S0, Dem_B_S1, Dem_B_S2, Dem_B_DoP, Dem_B_AoP] ...
    = Process_images_stokes(Dem_0(:,:,3), Dem_45(:,:,3), Dem_90(:,:,3), Dem_135(:,:,3), A);

% save demosaicked images
imwrite(Dem_90,sprintf('Results_color/Dem_90.png'));
imwrite(Dem_45,sprintf('Results_color/Dem_45.png'));
imwrite(Dem_135,sprintf('Results_color/Dem_135.png'));
imwrite(Dem_0,sprintf('Results_color/Dem_0.png'));
Dem_S0 = cat(3,Dem_R_S0,Dem_G_S0,Dem_B_S0);
imwrite(Dem_S0,sprintf('Results_color/Dem_S0.png'));

imwrite(Dem_S0,sprintf('Results_color/Dem_R_S0.png'));
imwrite(aolp_dolp(Dem_R_AoP,sqrt(Dem_R_DoP)),sprintf('Results_color/Dem_R_AoP_DoP.png'));

imwrite(Dem_G_S0,sprintf('Results_color/Dem_G_S0.png'));
imwrite(aolp_dolp(Dem_G_AoP,sqrt(Dem_G_DoP)),sprintf('Results_color/Dem_G_AoP_DoP.png'));

imwrite(Dem_B_S0,sprintf('Results_color/Dem_B_S0.png'));
imwrite(aolp_dolp(Dem_B_AoP,sqrt(Dem_B_DoP)),sprintf('Results_color/Dem_B_AoP_DoP.png'));

%% CPSNR calculation             
S1 = cat(3,R_S1,G_S1,B_S1);       
S2 = cat(3,R_S2,G_S2,B_S2);       
DoP = cat(3,R_DoP,G_DoP,B_DoP);       
AoP = cat(3,R_AoP,G_AoP,B_AoP);               
Dem_S1 = cat(3,Dem_R_S1,Dem_G_S1,Dem_B_S1);     
Dem_S2 = cat(3,Dem_R_S2,Dem_G_S2,Dem_B_S2);     
Dem_DoP = cat(3,Dem_R_DoP,Dem_G_DoP,Dem_B_DoP);    
Dem_AoP = cat(3,Dem_R_AoP,Dem_G_AoP,Dem_B_AoP);
              
cpsnr_90 = imcpsnr(GT_90,Dem_90,1,15);        
cpsnr_45 = imcpsnr(GT_45,Dem_45,1,15);        
cpsnr_135 = imcpsnr(GT_135,Dem_135,1,15);         
cpsnr_0 = imcpsnr(GT_0,Dem_0,1,15);
    
cpsnr_S0 = imcpsnr(S0,Dem_S0,1,15);        
cpsnr_S1 = imcpsnr(S1,Dem_S1,1,15);      
cpsnr_S2 = imcpsnr(S2,Dem_S2,1,15);       
cpsnr_DOLP = imcpsnr(DoP,Dem_DoP,1,15);      
angleerror = angleerror_AOLP(AoP,Dem_AoP,15);       

result = [cpsnr_0,cpsnr_45,cpsnr_90,cpsnr_135,cpsnr_S0,cpsnr_S1,cpsnr_S2,cpsnr_DOLP,angleerror]
csvwrite('Results_color/color.csv',result);