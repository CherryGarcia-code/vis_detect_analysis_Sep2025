
cd('/mnt/andreik/winstor/swc/mrsic_flogel/public/projects/AnKh_20200820_NPX_DMDM/Temporal expectation data/');

subjects = dir();
subjects(contains({subjects.name}, '.')) = [];
% subjects([1 2 4 6 12]) = [];
sus_sessions = [];

for i = 1:length(subjects)
   sus_sessions = [ sus_sessions  checkRecordingsSyncSubj(fullfile(subjects(i).folder, subjects(i).name)) ];
end