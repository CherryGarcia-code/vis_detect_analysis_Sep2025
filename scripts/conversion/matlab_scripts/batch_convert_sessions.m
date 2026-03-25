% batch_convert_sessions.m
% This script iterates over session folders, runs loadSessionNPX_main, and saves the result.

% ---------------- Configuration ----------------
% Set the directory containing the session folders you want to process
% Example: 'X:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\Raw data\'
sessions_root_dir = 'PATH_TO_YOUR_SESSIONS_ROOT_FOLDER'; 

% Set the output directory where .mat files will be saved
% Example: 'E:\python_analysis\git_repos\vis_detect_analysis_Sep2025\data\mat'
output_base_dir = 'PATH_TO_OUTPUT_DIRECTORY'; 
% -----------------------------------------------

% Get list of all subdirectories in the root directory
items = dir(sessions_root_dir);
is_dir = [items.isdir];
subdirs = items(is_dir);
subdir_names = {subdirs.name};
% Filter out '.' and '..'
subdir_names = subdir_names(~ismember(subdir_names, {'.', '..'}));

fprintf('Found %d session folders in %s\n', length(subdir_names), sessions_root_dir);

for i = 1:length(subdir_names)
    session_folder_name = subdir_names{i};
    session_full_path = fullfile(sessions_root_dir, session_folder_name);
    
    fprintf('--------------------------------------------------\n');
    fprintf('Processing session %d/%d: %s\n', i, length(subdir_names), session_folder_name);
    
    try
        % Call the loading function
        % Ensure loadSessionNPX_main is in your MATLAB path
        [data, subject_name, session_name] = loadSessionNPX_main(session_full_path);
        
        % Determine output path
        % Structure: output_base_dir / subject_name / subject_name_session_name.mat
        
        % Create subject directory if it doesn't exist
        subject_out_dir = fullfile(output_base_dir, subject_name);
        if ~exist(subject_out_dir, 'dir')
            mkdir(subject_out_dir);
        end
        
        % Construct filename
        % Check if session_name already contains subject_name to avoid duplication
        if startsWith(session_name, subject_name)
             filename = sprintf('%s.mat', session_name);
        else
             filename = sprintf('%s_%s.mat', subject_name, session_name);
        end
        
        save_path = fullfile(subject_out_dir, filename);
        
        fprintf('Saving data to %s...\n', save_path);
        save(save_path, 'data', '-v7.3'); % Use v7.3 for large files
        fprintf('Done.\n');
        
    catch ME
        fprintf('FAILED to process %s\n', session_folder_name);
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for k = 1:length(ME.stack)
            fprintf('  File: %s, Name: %s, Line: %d\n', ME.stack(k).file, ME.stack(k).name, ME.stack(k).line);
        end
    end
end

fprintf('--------------------------------------------------\n');
fprintf('Batch processing complete.\n');
