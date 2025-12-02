clear;

% % settings addpath(genpath('Functions'));
addpath(genpath('BM3D'));

% Result folder if exist ('Results_eval') == 0 mkdir('Results_eval') end

                                                 % Stokes Parameters matrix A =
    [ 1, 1, 0, 0; 1, 0, 1, 0; 1, -1, 0, 0; 1, 0, -1, 0 ];

% Get all subdirectories in results folder results_dir = 'results';
subdirs = dir(results_dir);
subdirs = subdirs([subdirs.isdir] & ~strncmp({subdirs.name}, '.', 1));

% Process each subdirectory
for subdir_idx = 1:length(subdirs)
    subdir_name = subdirs(subdir_idx).name;
subdir_path = fullfile(results_dir, subdir_name);

fprintf('Processing folder: %s\n', subdir_name);

% Get all Scene directories scene_dirs = dir(subdir_path);
scene_dirs =
    scene_dirs([scene_dirs.isdir] & ~strncmp({scene_dirs.name}, '.', 1));

all_results = [];
scene_names_list = {};

    % Process each Scene directory
    for scene_idx = 1:length(scene_dirs)
        scene_name = scene_dirs(scene_idx).name;
    scene_path = fullfile(subdir_path, scene_name);

    fprintf('  Processing scene: %s\n', scene_name);

    % Load GT images GT_0_path = fullfile(scene_path, 'GT_0.png');
    GT_45_path = fullfile(scene_path, 'GT_45.png');
    GT_90_path = fullfile(scene_path, 'GT_90.png');
    GT_135_path = fullfile(scene_path, 'GT_135.png');

    % Load Dem images Dem_0_path = fullfile(scene_path, 'Dem_0.png');
    Dem_45_path = fullfile(scene_path, 'Dem_45.png');
    Dem_90_path = fullfile(scene_path, 'Dem_90.png');
    Dem_135_path = fullfile(scene_path, 'Dem_135.png');

    % Check if all files exist if ~exist(GT_0_path, 'file') ||
        ~exist(GT_45_path, 'file') || ... ~exist(GT_90_path, 'file') ||
        ~exist(GT_135_path, 'file') || ... ~exist(Dem_0_path, 'file') ||
        ~exist(Dem_45_path, 'file') || ... ~exist(Dem_90_path, 'file') ||
        ~exist(Dem_135_path, 'file')
            fprintf('    Warning: Missing files in %s, skipping...\n',
                    scene_name);
    continue;
    end

        % Load images GT_0 = imread(GT_0_path);
    GT_45 = imread(GT_45_path);
    GT_90 = imread(GT_90_path);
    GT_135 = imread(GT_135_path);

    Dem_0 = imread(Dem_0_path);
    Dem_45 = imread(Dem_45_path);
    Dem_90 = imread(Dem_90_path);
    Dem_135 = imread(Dem_135_path);

    % Convert to double if needed if ~isa(GT_0, 'double') GT_0 = double(GT_0) /
                                                                 255.0;
    GT_45 = double(GT_45) / 255.0;
    GT_90 = double(GT_90) / 255.0;
    GT_135 = double(GT_135) / 255.0;
    end

        if ~isa(Dem_0, 'double') Dem_0 = double(Dem_0) / 255.0;
    Dem_45 = double(Dem_45) / 255.0;
    Dem_90 = double(Dem_90) / 255.0;
    Dem_135 = double(Dem_135) / 255.0;
    end

        % %
        Calculate Stokes
        Parameters(Ground Truth)[R_S0, R_S1, R_S2, R_DoP, R_AoP] =
        Process_images_stokes(GT_0( :, :, 1), GT_45( :, :, 1), GT_90( :, :, 1),
                              GT_135( :, :, 1), A);
    [ G_S0, G_S1, G_S2, G_DoP, G_AoP ] = Process_images_stokes(
        GT_0( :, :, 2), GT_45( :, :, 2), GT_90( :, :, 2), GT_135( :, :, 2), A);
    [ B_S0, B_S1, B_S2, B_DoP, B_AoP ] = Process_images_stokes(
        GT_0( :, :, 3), GT_45( :, :, 3), GT_90( :, :, 3), GT_135( :, :, 3), A);

    % %
        Stokes Parameters(Demosaicked)[Dem_R_S0, Dem_R_S1, Dem_R_S2, Dem_R_DoP,
                                       Dem_R_AoP]... =
        Process_images_stokes(Dem_0( :, :, 1), Dem_45( :, :, 1),
                              Dem_90( :, :, 1), Dem_135( :, :, 1), A);
    [ Dem_G_S0, Dem_G_S1, Dem_G_S2, Dem_G_DoP, Dem_G_AoP ]... =
        Process_images_stokes(Dem_0( :, :, 2), Dem_45( :, :, 2),
                              Dem_90( :, :, 2), Dem_135( :, :, 2), A);
    [ Dem_B_S0, Dem_B_S1, Dem_B_S2, Dem_B_DoP, Dem_B_AoP ]... =
        Process_images_stokes(Dem_0( :, :, 3), Dem_45( :, :, 3),
                              Dem_90( :, :, 3), Dem_135( :, :, 3), A);

    % % CPSNR calculation S0 = cat(3, R_S0, G_S0, B_S0);
    S1 = cat(3, R_S1, G_S1, B_S1);
    S2 = cat(3, R_S2, G_S2, B_S2);
    DoP = cat(3, R_DoP, G_DoP, B_DoP);
    AoP = cat(3, R_AoP, G_AoP, B_AoP);

    Dem_S0 = cat(3, Dem_R_S0, Dem_G_S0, Dem_B_S0);
    Dem_S1 = cat(3, Dem_R_S1, Dem_G_S1, Dem_B_S1);
    Dem_S2 = cat(3, Dem_R_S2, Dem_G_S2, Dem_B_S2);
    Dem_DoP = cat(3, Dem_R_DoP, Dem_G_DoP, Dem_B_DoP);
    Dem_AoP = cat(3, Dem_R_AoP, Dem_G_AoP, Dem_B_AoP);

        %% Save AoP-DoP images
        % Create output directory for this method and scene
        output_dir = fullfile('Results_eval', subdir_name, scene_name);
        if
          ~exist(output_dir, 'dir') mkdir(output_dir);
        end

                % Save GT AoP -
            DoP images(RGB channels)
                imwrite(aolp_dolp(R_AoP, sqrt(R_DoP)),
                        fullfile(output_dir, 'GT_R_AoP_DoP.png'));
        imwrite(aolp_dolp(G_AoP, sqrt(G_DoP)),
                fullfile(output_dir, 'GT_G_AoP_DoP.png'));
        imwrite(aolp_dolp(B_AoP, sqrt(B_DoP)),
                fullfile(output_dir, 'GT_B_AoP_DoP.png'));

        % Save Dem AoP - DoP images(RGB channels)
                             imwrite(aolp_dolp(Dem_R_AoP, sqrt(Dem_R_DoP)),
                                     fullfile(output_dir, 'Dem_R_AoP_DoP.png'));
        imwrite(aolp_dolp(Dem_G_AoP, sqrt(Dem_G_DoP)),
                fullfile(output_dir, 'Dem_G_AoP_DoP.png'));
        imwrite(aolp_dolp(Dem_B_AoP, sqrt(Dem_B_DoP)),
                fullfile(output_dir, 'Dem_B_AoP_DoP.png'));

        cpsnr_90 = imcpsnr(GT_90, Dem_90, 1, 15);
        cpsnr_45 = imcpsnr(GT_45, Dem_45, 1, 15);
        cpsnr_135 = imcpsnr(GT_135, Dem_135, 1, 15);
        cpsnr_0 = imcpsnr(GT_0, Dem_0, 1, 15);

        cpsnr_S0 = imcpsnr(S0, Dem_S0, 1, 15);
        cpsnr_S1 = imcpsnr(S1, Dem_S1, 1, 15);
        cpsnr_S2 = imcpsnr(S2, Dem_S2, 1, 15);
        cpsnr_DOLP = imcpsnr(DoP, Dem_DoP, 1, 15);
        angleerror = angleerror_AOLP(AoP, Dem_AoP, 15);

        % SSIM calculation ssim_90 = mean(imssim(GT_90, Dem_90, 1, 15));
        ssim_45 = mean(imssim(GT_45, Dem_45, 1, 15));
        ssim_135 = mean(imssim(GT_135, Dem_135, 1, 15));
        ssim_0 = mean(imssim(GT_0, Dem_0, 1, 15));

        ssim_S0 = mean(imssim(S0, Dem_S0, 1, 15));
        ssim_S1 = mean(imssim(S1, Dem_S1, 1, 15));
        ssim_S2 = mean(imssim(S2, Dem_S2, 1, 15));
        ssim_DOLP = mean(imssim(DoP, Dem_DoP, 1, 15));

        result = [
          cpsnr_0, cpsnr_45, cpsnr_90, cpsnr_135, cpsnr_S0, cpsnr_S1, cpsnr_S2,
          cpsnr_DOLP, angleerror, ... ssim_0, ssim_45, ssim_90, ssim_135,
          ssim_S0, ssim_S1, ssim_S2,
          ssim_DOLP
        ];
        all_results = [all_results; result];
        scene_names_list{end + 1} = scene_name;

        fprintf(
            '    Scene %s: CPSNR_0=%.2f, CPSNR_45=%.2f, CPSNR_90=%.2f, CPSNR_135=%.2f\n',
            ... scene_name, cpsnr_0, cpsnr_45, cpsnr_90, cpsnr_135);
    end
    
    % Save results for this subdirectory
    if ~isempty(all_results)
        avg_result = mean(all_results, 1);
    csv_file = sprintf('Results_eval/%s.csv', subdir_name);

    % Headers headers = {
        'Scene',        'CPSNR_0',  'CPSNR_45', 'CPSNR_90',   'CPSNR_135',
        ... 'CPSNR_S0', 'CPSNR_S1', 'CPSNR_S2', 'CPSNR_DOLP', 'AngleError',
        ... 'SSIM_0',   'SSIM_45',  'SSIM_90',  'SSIM_135',   ... 'SSIM_S0',
        'SSIM_S1',      'SSIM_S2',  'SSIM_DOLP'};

        % Open file for writing
        fid = fopen(csv_file, 'w');

        % Write headers fprintf(fid, '%s,', headers{1 : end - 1});
        fprintf(fid, '%s\n', headers{end});

        % Write data rows
        for i = 1:size(all_results, 1)
            fprintf(fid, '%s,', scene_names_list{i});
        fprintf(fid, '%.4f,', all_results(i, 1 : end - 1));
        fprintf(fid, '%.4f\n', all_results(i, end));
        end

            % Write average row fprintf(fid, 'Average,');
        fprintf(fid, '%.4f,', avg_result(1 : end - 1));
        fprintf(fid, '%.4f\n', avg_result(end));

        fclose(fid);

        fprintf('  Average results saved to %s\n', csv_file);
        fprintf(
            '  Average: CPSNR_0=%.2f, CPSNR_45=%.2f, CPSNR_90=%.2f, CPSNR_135=%.2f\n',
            ... avg_result(1), avg_result(2), avg_result(3), avg_result(4));
        end end

            fprintf('Evaluation completed!\n');
