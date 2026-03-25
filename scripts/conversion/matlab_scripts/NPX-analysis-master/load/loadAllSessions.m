function data = loadAllSessions(varargin)

if ~isempty(varargin)
    data = varargin{1};
else
    data = [];
end

try
    cd('/mnt/andreik/winstor/swc/mrsic_flogel/public/projects/AnKh_20200820_NPX_DMDM/Temporal expectation data/');
end
subject_path = uigetdir(pwd, 'Select subject folder');

cd(subject_path);
cd('Raw data');

sessions = dir(pwd); 
sessions(contains({sessions.name}, '.')) = [];

for i = 1:length(sessions)
    session_path = fullfile(subject_path, 'Raw data', sessions(i).name);
    [NPX_data_sesion, subject_name, session_name] = loadSessionNPX_main(session_path);
    data.(subject_name).(session_name) = NPX_data_sesion.(subject_name).(session_name);
end

end