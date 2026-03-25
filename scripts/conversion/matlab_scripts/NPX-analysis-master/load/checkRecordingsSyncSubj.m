function sus_sessions = checkRecordingsSyncSubj(varargin)

%CHECKRECORINGSYNCSTARTEND 
% Check if recodings of imec(s) and nidaq streams was started simultaneusly, show sessions with unmatched number of SYNC singal periods if found  
% Should be run after TPrime

if isempty(varargin)
    try
        cd('/mnt/andreik/winstor/swc/mrsic_flogel/public/projects/AnKh_20200820_NPX_DMDM/Temporal expectation data/');
    end
    subject_path = uigetdir(pwd, 'Select subject folder');
else
    subject_path = varargin{1}; % supply subject path directly 
end

sessions_processed = dir(fullfile(subject_path, 'Processed data'));
sessions_processed(contains({sessions_processed.name}, '.')) = [];
sessions_processed(contains({sessions_processed.name}, 'Histology')) = [];

sus_sessions_count = 0;
for i = 1:length(sessions_processed)
    
    ses_processed_data_path = fullfile(sessions_processed(i).folder, sessions_processed(i).name);
    cd(fullfile(ses_processed_data_path, 'Nidaq'));

    fid_NI_sync = fopen(fullfile(ses_processed_data_path, 'Nidaq', 'NI_Sync.txt'), 'r');
    NI_sync  = fscanf(fid_NI_sync,'%f');
    fclose(fid_NI_sync);
    
    cd(fullfile(ses_processed_data_path, 'Kilosort&Phy'));
    probe_folders = dir(pwd);
    probe_folders(contains({probe_folders.name}, '.')) = [];
    probes_numb = length(probe_folders);
    
    for p = 1:probes_numb
        probe_path = fullfile(probe_folders(p).folder, probe_folders(p).name);
        cd(probe_path);
        probe_sync_txt_path_tmp = dir('*tcat.imec*.txt');

        fid_probe_sync = fopen(fullfile(probe_sync_txt_path_tmp.folder, probe_sync_txt_path_tmp.name), 'r');
        probe_sync  = fscanf(fid_probe_sync,'%f');
        fclose(fid_probe_sync);
        probe_sync_length(p) = length(probe_sync);
        
        if abs(probe_sync_length(p) - length(NI_sync)) >=2      % 1 period discrepancy here is apparently ok and doesn't influence synchronization, likely happens because GatGT extracts only full periods of sync signal so if the last one is partial (only rising edge), it is not recorded  
            sus_sessions_count = sus_sessions_count+1;
            sus_sessions(sus_sessions_count).ses_name = probe_folders(p).name;
            sus_sessions(sus_sessions_count).ses_ind = i;
            sus_sessions(sus_sessions_count).probe_ind = p;
            sus_sessions(sus_sessions_count).sync_diff_prom_NI = probe_sync_length(p) - length(NI_sync);
            sus_sessions(sus_sessions_count).sync_diff_p2p = [];
        end
    end
    
    if probes_numb>1
        if abs(probe_sync_length(1) - probe_sync_length(2)) >0
            sus_sessions_count = sus_sessions_count+1;
            sus_sessions(sus_sessions_count).ses_name = probe_folders(p).name;
            sus_sessions(sus_sessions_count).ses_ind = i;
            sus_sessions(sus_sessions_count).probe_ind = [];
            sus_sessions(sus_sessions_count).sync_diff_prom_NI = [];
            sus_sessions(sus_sessions_count).sync_diff_p2p = probe_sync_length(1) - probe_sync_length(2);
        end
    end
        
end   
if sus_sessions_count == 0 
    sus_sessions = [];
end

end

