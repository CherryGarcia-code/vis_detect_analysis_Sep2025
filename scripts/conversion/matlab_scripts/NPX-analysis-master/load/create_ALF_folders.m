


cd('/mnt/andreik/winstor/swc/mrsic_flogel/public/projects/AnKh_20200820_NPX_DMDM/Temporal expectation data/');
subject_path = uigetdir(pwd, 'Select subject folder');

cd(subject_path);
cd('Processed data');

sessions = dir(pwd); 
sessions(contains({sessions.name}, '.')) = [];
sessions(contains({sessions.name}, 'Histology')) = [];

for i = 1:length(sessions)
    session_path = fullfile(subject_path, 'Processed data', sessions(i).name);
    cd(fullfile(session_path, 'Kilosort&Phy'));
    
    probe_folders = dir(pwd); 
    probe_folders(contains({probe_folders.name}, '.')) = [];
    
    cd(session_path);
    mkdir('IBL_ALF');
    cd('IBL_ALF');
    
    for j = 1:length(probe_folders)
        mkdir(probe_folders(j).name);
    end
    
end

