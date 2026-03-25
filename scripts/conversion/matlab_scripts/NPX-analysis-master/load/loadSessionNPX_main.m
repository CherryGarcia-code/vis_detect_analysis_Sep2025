
function [data, subject_name, session_name] = loadSessionNPX_main(varargin)

    currDir = pwd;
    if isempty(varargin)
        try
            cd('X:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\Raw data\');
        end
        session_path = uigetdir(pwd, 'Select session folder with raw data');
    else
        session_path = varargin{1};
    end
    
    ind_tmp = strfind(session_path, '\');
    subject_name = session_path( ind_tmp(end-2)+1:ind_tmp(end-1)-1 );
    subject_name = genvarname(subject_name);
    subject_path = session_path( 1:ind_tmp(end-1) );
    session_name = session_path( ind_tmp(end)+1:end );

    % load beahvioral data
    behav_data = loadSessionBehav(session_path);
%     behav_data = loadSessionBehavMorioFormat(session_path);

    % load NIdaq events 
    session_path = fullfile(subject_path, 'Processed data', session_name);
    cd(session_path);
    cd('Nidaq');
    NI_events_file = dir( '*.mat');
    NI_events = load(NI_events_file.name);
    NI_events = NI_events.NIdaq_events;
    
    % load videography analysis 

    try
        Video = loadSessionVideo(session_path);
    catch
        Video = [];
    end

    % load Kilosort data, keep only units that were labeled "good" for first
    % pass analysis

    cd('..');
    cd('Kilosort&Phy');
    probe_folders = dir();
    probe_folders(contains({probe_folders.name}, '.')) = [];
    probe_folders(contains({probe_folders.name}, 'Sorted')) = [];
    
    for p = 1:length(probe_folders)
        kilosort_path = fullfile(session_path, 'Kilosort&Phy', probe_folders(p).name);

        ALF_path = fullfile(session_path, 'IBL_ALF', probe_folders(p).name);
%         ALF_path = fullfile(session_path, 'AE_GUI_Ephys', probe_folders(p).name);

        cd(kilosort_path);
        probe = loadKSdir(pwd);
        cluster_qual_KS_tmp = tdfread( [pwd '/cluster_KSLabel.tsv'] , 'tab');

        is_cl_labeled_good = [];
        for cl = 1:length(cluster_qual_KS_tmp.cluster_id)
            is_cl_labeled_good(cl) = strcmp(cluster_qual_KS_tmp.KSLabel(cl,:), 'good');
        end
        probe.cluster_id_KS_good = cluster_qual_KS_tmp.cluster_id(is_cl_labeled_good==1);
        probe.cluster_id_good_and_stable = find_good_stable_units_PaperVersion(probe);
        
        
        try
            % read probe location in Allen CCF coordinates using priviously done track tracing 
            probe_coord = getProbe_location(ALF_path);
            probe.probe_coord = probe_coord;

            % calculate channel with max aplitude for each good unit, get corresponding brain area
            [~, good_cl_ind, ~] = intersect(probe.cids, probe.cluster_id_KS_good);
            [~, max_ch_good_cl] = max(max(abs(probe.temps(good_cl_ind,:,:)), [], 2), [], 3);

            clearvars good_cl_coord;
            for cl = 1:length(good_cl_ind)
                good_cl_coord(cl) = probe_coord(max_ch_good_cl(cl));
            end 
            good_cl_coord = rmfield(good_cl_coord ,'axial');
            good_cl_coord = rmfield(good_cl_coord ,'lateral');

            probe.good_cl_coord = good_cl_coord;
            
            [~, good_and_st_cl_ind, ~ ] = intersect(probe.cluster_id_KS_good, probe.cluster_id_good_and_stable);
            good_and_stab_cl_coord = good_cl_coord(good_and_st_cl_ind);
            probe.good_and_stab_cl_coord = good_and_stab_cl_coord;
        end
        
        [spikeAmps, spikeDepths, templateDepths, tempAmps, tempsUnW, templateDuration, waveforms] = templatePositionsAmplitudes(probe.temps, probe.winv, probe.ycoords, probe.spikeTemplates, probe.tempScalingAmps);
        probe.templateDepths = templateDepths;
        probe.templateWaveforms = waveforms;
        
        probe = trim_probe_struc(probe);
        data.(subject_name).(session_name).NPX_probes(p) = probe;
        
    end
    
    data.(subject_name).(session_name).NI_events = NI_events;
    data.(subject_name).(session_name).Video = Video;
    data.(subject_name).(session_name).behav_data = behav_data;
    
    cd(currDir)
end


function probe = trim_probe_struc(probe)

%     good_clusters = probe.cluster_id_KS_good;
    good_and_stable_clusters = probe.cluster_id_good_and_stable;
    all_clusters_series = probe.clu;
    st = probe.st;
    all_clusters = unique(all_clusters_series);

    % leave spike times only from good and stable units
    ind_to_del = [];
    for cl = 1:length(all_clusters)
        if ~ismember( all_clusters(cl), good_and_stable_clusters)
            ind_to_del = [ind_to_del ; find(all_clusters_series == all_clusters(cl))];
        end
    end

    all_clusters_series(ind_to_del) = [];
    st(ind_to_del) = [];
    probe.clu = all_clusters_series;
    probe.st = st;
    probe = rmfield(probe, {'spikeTemplates', 'tempScalingAmps', 'temps', 'winv'});
end

            