

PSTHwindow = [-0.3 1.2];
Stim1Ori_use = [ 90 270]; % drift direction(s) of gratings 
subjects = fieldnames(data);
TrialGroups = [];
EventTimes = [];
sigma = 0.025;
colors = lines(6);
colors = [[0 0 0] ; colors];
binSize = 0.001;

for i = 1 % which mouse
    
    sessions = fieldnames(data.(subjects{i}));
    for j = 1 % which session
        
        probes_numb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        
        Baseline_ON_times = data.(subjects{i}).(sessions{j}).NI_events.Baseline_ON.rise_t;
        Change_ON_times = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.rise_t;
        Change_ON_dur = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.duration;
        
        Reward_times = data.(subjects{i}).(sessions{j}).NI_events.Valve_L.rise_t;
        Airpuff_times = data.(subjects{i}).(sessions{j}).NI_events.Air_puff.rise_t;
        frame_times = data.(subjects{i}).(sessions{j}).NI_events.frame_times_tr.time;
%         MotionOnsetTimes = data.(subjects{i}).(sessions{j}).Video.MotionOnsetTimes;
        
        Change_ON_times_new = [];
        for tr = 1:length(Change_ON_times)
            if isnan(Change_ON_times(tr))
                Change_ON_times_new(tr) = NaN;
            else
                Change_ON_times_new(tr) = frame_times{tr}(find(frame_times{tr} >= Change_ON_times(tr), 1, 'first'));
            end
        end
        
        if nansum( (Change_ON_times_new - Change_ON_times)>0.05)>0
            ind = find((Change_ON_times_new - Change_ON_times)>0.05);
            disp(Change_ON_times_new(ind)-Change_ON_times(ind))
        else
            Change_ON_times = Change_ON_times_new;
        end
                
        trials_numb  = length(Baseline_ON_times);
        TrialsData = data.(subjects{i}).(sessions{j}).behav_data.trials_data_exp;
        
        ReactionTimes = [TrialsData.reactiontimes];
        ReactionTimesFA = [ReactionTimes.FA];
        ReactionTimesHits = [ReactionTimes.RT];
        ReactionTimesAbort = [ReactionTimes.abort];
        AbortTimes = Baseline_ON_times+ReactionTimesAbort;

        Change_magn = [TrialsData.Stim2TF];
        Stim1Ori = [TrialsData.Stim1Ori];
        TF = {TrialsData.TF};
        phase = {TrialsData.phase};
        
        blockTypeNoise = [TrialsData.blockTypeNoise];
        trialsd = [TrialsData.trialsd];
        
        try
            isLaserOn = [TrialsData.LaserOn];
        end
        
        picked_Stim1Ori_trials = sum((Stim1Ori==Stim1Ori_use'),1);
        hit_trials = ([TrialsData.IsHit]==1);
        early_blocks_hit_trials = ([TrialsData.IsEarlyBlock]==1&[TrialsData.IsProbe]==0) & ([TrialsData.IsHit]==1) & (picked_Stim1Ori_trials==1);
        early_blocks_miss_trials = (( ([TrialsData.IsEarlyBlock]==1)&([TrialsData.IsProbe]==0) ) ) & ([TrialsData.IsMiss]==1) & (picked_Stim1Ori_trials==1);

        late_blocks_hit_trials = ( ([TrialsData.IsLateBlock]==1)&([TrialsData.IsProbe]==0) )  & ([TrialsData.IsHit]==1) & (picked_Stim1Ori_trials==1);
        late_blocks_miss_trials = (( ([TrialsData.IsLateBlock]==1)&([TrialsData.IsProbe]==0) ) ) & ([TrialsData.IsMiss]==1) & (picked_Stim1Ori_trials==1);
        
        TrialGroupsNames = [];

%          TF_incr_frame_times = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.5, 10, 'log2', PSTHwindow(2), find(blockTypeNoise==2&trialsd==0.25));
%          TF_decr_frame_times = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, -10, -1.5, 'log2', PSTHwindow(2), find(blockTypeNoise==2&trialsd==0.25));

         TF_incr_frame_times = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.7, 10, 'lin', PSTHwindow(2), 1:trials_numb);
         TF_decr_frame_times = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0, 0.6, 'lin', PSTHwindow(2), 1:trials_numb);         
%          
%          TF_incr_incr_frame_times = get_TF_pulses_seq(TrialsData, Change_ON_dur, frame_times,{[1.5,10], [1.5,10]}, 0.05, 'log2', PSTHwindow(2), find(picked_Stim1Ori_trials==1));
%          TF_incr_decr_frame_times = get_TF_pulses_seq(TrialsData, Change_ON_dur, frame_times,{[-10,-1.5], [-10,-1.5]}, 0.05, 'log2', PSTHwindow(2), find(picked_Stim1Ori_trials==1));

%          TF_incr_frame_times = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.15, 2, 'lin', PSTHwindow(2), find(picked_Stim1Ori_trials==1));
%          TF_decr_frame_times = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0.5, 0.85, 'lin', PSTHwindow(2), find(picked_Stim1Ori_trials==1));
%          TF_incr_frame_times2 = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.15, 1.25, 'lin', PSTHwindow(2), find(picked_Stim1Ori_trials==1));
%          TF_decr_frame_times2 = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0.85, 0.95, 'lin', PSTHwindow(2), find(picked_Stim1Ori_trials==1));

%           [TF_incr_incr_frame_times, TF_incr_incr_frame_ind] = get_TF_pulses_seq(TrialsData, Change_ON_dur, frame_times,{[1.25,2], [1.25,2]}, 0.45, 'lin', PSTHwindow(2), find(picked_Stim1Ori_trials==1));
%          TF_incr_decr_frame_times = get_TF_pulses_seq(TrialsData, Change_ON_dur, frame_times,{[0.5,0.85], [0.5,0.85]}, 0.05, 'lin', PSTHwindow(2), find(picked_Stim1Ori_trials==1));    
%                  
%          TF_incr_frame_timeEarly = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.25, 10, 'lin', PSTHwindow(2), find([TrialsData.IsEarlyBlock]==1), [2 6], 2, 1, 0.0167);
%          TF_decr_frame_timesEarly = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0.5, 0.8, 'lin', PSTHwindow(2), find([TrialsData.IsEarlyBlock]==1), [2 6], 2, 1, 0.0167);
%          TF_incr_frame_timesLate = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.25, 10, 'lin', PSTHwindow(2), find([TrialsData.IsLateBlock]==1), [2 6], 2, 1, 0.0167);
%          TF_decr_frame_timesLate = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0.5, 0.8, 'lin', PSTHwindow(2), find([TrialsData.IsLateBlock]==1), [2 6], 2, 1, 0.0167);

 
% 
%          TF_incr_frame_timesUP = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.15, 2, 'lin', PSTHwindow(2), find(Stim1Ori==Stim1Ori_use(1)));
%          TF_decr_frame_timesUP = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0.5, 0.85, 'lin', PSTHwindow(2), find(Stim1Ori==Stim1Ori_use(1)));
%          TF_incr_frame_timesDown = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.15, 2, 'lin', PSTHwindow(2), find(Stim1Ori==Stim1Ori_use(2)));
%          TF_decr_frame_timesDown = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0.5, 0.85, 'lin', PSTHwindow(2), find(Stim1Ori==Stim1Ori_use(2)));


%      TF_incr_frame_timesLaserOff = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.15, 2, 'lin', PSTHwindow(2), find(isLaserOn==0));
%      TF_decr_frame_timesLaserOff = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0.5, 0.85, 'lin', PSTHwindow(2), find(isLaserOn==0));
%      TF_incr_frame_timesLaserOn = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 1.15, 2, 'lin', PSTHwindow(2), find(isLaserOn==1));
%      TF_decr_frame_timesLaserOn = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, 0.5, 0.85, 'lin', PSTHwindow(2), find(isLaserOn==1));
                  
%% aligned to Baseline
% % % % 
%         EventTimes = Baseline_ON_times;
%         TrialGroups = zeros(1, trials_numb);
% % %         TrialGroups(hit_trials==1&Stim1Ori==90) = 1;
% % %         TrialGroups(hit_trials==1&Stim1Ori==270) = 2;
%         TrialGroups(Stim1Ori==90) = 1;
%         TrialGroups(Stim1Ori==270) = 2;
%         TrialGroupsNames{1} = 'Upwards drift';
%         TrialGroupsNames{2} = 'Downwards drift'; 

%         TrialGroups(isLaserOn==0&[TrialsData.IsMiss]==1) = 1;
%         TrialGroups(isLaserOn==1&[TrialsData.IsMiss]==1) = 2;

%         TrialGroups(isLaserOn==0&early_blocks_hit_trials==1) = 1;
%         TrialGroups(isLaserOn==1&early_blocks_hit_trials==1) = 2;

% % %         
% %         EventTimes = Baseline_ON_times;
%         TrialGroups = zeros(1, trials_numb);

%         TrialGroups(early_blocks_hit_trials==1) = 1;
%         TrialGroups(late_blocks_hit_trials==1) = 2;
        
%         TrialGroups(early_blocks_miss_trials==1) = 2;
%         TrialGroups(late_blocks_miss_trials==1) = 2;
        
%         TrialGroupsNames{1} = 'Hits, early';
%         TrialGroupsNames{2} = 'Hits, late';
%         TrialGroupsNames{2} = 'Misses, early';
%         TrialGroupsNames{4} = 'Misses, late';        
%         
%% aligned to early licks
% 
%         EventTimes = Airpuff_times;
%         TrialGroups = zeros(1, trials_numb);
%         early_lick_trials = (ReactionTimesFA >2); % exclude impulsive licks
%         TrialGroups(early_lick_trials) = 1;
%         TrialGroups(early_lick_trials&blockTypeNoise==1) = 1;
%         TrialGroups(early_lick_trials&blockTypeNoise==2) = 2;
        
%         TrialGroups(early_lick_trials&isLaserOn==0) = 1;
%         TrialGroups(early_lick_trials&isLaserOn==1) = 2;
%% align to aborts

%         EventTimes = AbortTimes;
%         TrialGroups = zeros(1, trials_numb);
%         abortTrials = (ReactionTimesAbort>2); % exclude too early aborts
%         TrialGroups(abortTrials&isLaserOn==0) = 1;
%         TrialGroups(abortTrials&isLaserOn==1) = 2;

%% aligned to TF pulse increase vs decrease
        EventTimes{1} = cell2mat(TF_incr_frame_times(:)');
%         EventTimes{2} = cell2mat(TF_incr_incr_frame_times(:)');

%         EventTimes{2} = cell2mat(TF_incr_frame_times2(:)');
%         EventTimes{3} = cell2mat(TF_decr_frame_times2(:)');
        EventTimes{2} = cell2mat(TF_decr_frame_times(:)');
%         
        TrialGroupsNames{1} = 'Large TF incr';
%         TrialGroupsNames{2} = 'Small TF incr';
%         TrialGroupsNames{3} = 'Small TF decr';
        TrialGroupsNames{2} = 'Large TF decr';        
%          
%         EventTimes{1} = cell2mat(TF_incr_frame_timesLaserOff(:)');
%         EventTimes{2} = cell2mat(TF_incr_frame_timesLaserOn(:)');
%         EventTimes{3} = cell2mat(TF_decr_frame_timesLaserOff(:)');
%         EventTimes{4} = cell2mat(TF_decr_frame_timesLaserOn(:)');
%         
%         TrialGroupsNames{1} = 'TF incr';
%         TrialGroupsNames{2} = 'TF incr, laser';
%         TrialGroupsNames{3} = 'TF decr';
%         TrialGroupsNames{4} = 'TF decr, laser';  


%         EventTimes{1} = cell2mat(TF_incr_frame_timeEarly(:)');
%         EventTimes{2} = cell2mat(TF_incr_frame_timesLate(:)');
%         EventTimes{3} = cell2mat(TF_decr_frame_timesEarly(:)');
%         EventTimes{4} = cell2mat(TF_decr_frame_timesLate(:)');
%         
%         TrialGroupsNames{1} = 'Early TF incr';
%         TrialGroupsNames{2} = 'Late TF incr';
%         TrialGroupsNames{3} = 'Early TF decr';
%         TrialGroupsNames{4} = 'Late TF decr';  

%         EventTimes{1} = cell2mat(TF_incr_frame_timesUP(:)');
%         EventTimes{2} = cell2mat(TF_decr_frame_timesUP(:)');
%         EventTimes{3} = cell2mat(TF_incr_frame_timesDown(:)');
%         EventTimes{4} = cell2mat(TF_decr_frame_timesDown(:)');
%         
%         TrialGroupsNames{1} = 'TF incr Up';
%         TrialGroupsNames{2} = 'TF dect Up';
%         TrialGroupsNames{3} = 'TF incr Down';
%         TrialGroupsNames{4} = 'TF dect Down';      
% % 
        for ev = 1:length(EventTimes)
            TrialGroups = [TrialGroups ev*ones(1, length(EventTimes{ev}))];
        end
        EventTimes = cell2mat(EventTimes);
% %         
%         TrialGroupsNames{1} = 'TF incr';
%         TrialGroupsNames{2} = 'TF decr';
%         TrialGroupsNames{3} = 'TF incr->inrc';
%         TrialGroupsNames{4} = 'TF incr->decr';
%         EventTimes = cell2mat(EventTimes);
        

        %% aligned to TF pulse increase, parsed by phase at which TF pulse occured 
% 
%         EventTimes = [cell2mat(TF_incr_frame_times(:)')  cell2mat(TF_decr_frame_times(:)')];
%         tr_to_use = find(sum((Stim1Ori==Stim1Ori_use'), 1) == 1);      
%         TF_incr_frame_ind = TF_incr_frame_ind(tr_to_use);   
%         TF_incr_frame_times = TF_incr_frame_times(tr_to_use);     
%         phase = phase(tr_to_use);     
%        
%         EventTimes = [cell2mat(TF_incr_frame_times(:)')];
%         TrialGroups = zeros(1, length(EventTimes));
%         
%         phase_at_TF_increase = [];
%         for tr = 1:length(TF_incr_frame_times)
%             TF_incr_frame_ind_tr = TF_incr_frame_ind{tr};
%             phase_tr = phase{tr}(:,1);
%             ind = find(phase_tr>0, 1, 'first');
%             phase_tr = phase_tr(ind-1:end); % skip pre-baseline
%             phase_tr = mod(phase_tr, 360);
%             for fr = 1:length(TF_incr_frame_ind_tr)
%                 phase_at_TF_increase = [phase_at_TF_increase phase_tr(TF_incr_frame_ind_tr(fr))];
%             end
%         end
%          
%         phase_bin = 90;
%         phase_bins = 0:phase_bin:360;
%         for ph = 1:length(phase_bins)-1
%             phases_in_bin_ind = find(phase_at_TF_increase>=phase_bins(ph)&phase_at_TF_increase<phase_bins(ph+1));
%             TrialGroups(phases_in_bin_ind) = ph;
%         end

       
%% aligned to Change onset, parsed by Change magnitude

%         EventTimes = Change_ON_times;
%         EventTimes = MotionOnsetTimes;
%         TrialGroups = zeros(1, trials_numb);
%     
% %         TrialGroups(~isnan(EventTimes))=1;
%         
%         Change_magn_types = unique(Change_magn);
% %         
% %         Change_magn_types(Change_magn_types==1) = [];   % deleting no change trials
%  
%         for ch = 2:6
%             if ch ==1
%                 TrialGroups( (Change_magn==Change_magn_types(ch)) & ([TrialsData.IsMiss]==1)) = 1;
%             else
% %                 if ch < 5 % two groups of change strength
% %                     TrialGroups( (Change_magn==Change_magn_types(ch)) & ([TrialsData.IsHit]==1)) = 2;
% %                 else
% %                     TrialGroups( (Change_magn==Change_magn_types(ch)) & ([TrialsData.IsHit]==1) ) = 3;
% %                 end
%                 TrialGroups( (Change_magn==Change_magn_types(ch)) & ([TrialsData.IsHit]==1)) = ch;
%             end
%             TrialGroupsNames{ch-1} = [num2str(Change_magn_types(ch)) ' Hz Change'];
%         end
%         TrialGroups(isnan(EventTimes))=0;
%         
%         too_fast_RT = find(ReactionTimesHits < 0.6);
%         
%         too_fast_RT = find(ReactionTimesHits < PSTHwindow(2));
%         TrialGroups(too_fast_RT) = 0;
%%
%         TrialGroups(picked_Stim1Ori_trials==0) = 0;
        not_used_tr_ind = find(TrialGroups==0);
        TrialGroups(not_used_tr_ind) = [];
        EventTimes(not_used_tr_ind) = [];
                 
        for p = 1%which probe
            sp = data.(subjects{i}).(sessions{j}).NPX_probes(p);
            
%             good_and_stable_clusters = data.(subjects{i}).(sessions{j}).NPX_probes(p).cluster_id_KS_good;
            good_and_stable_clusters = data.(subjects{i}).(sessions{j}).NPX_probes(p).cluster_id_good_and_stable;  

            try     % if probe track tracing has been done
                good_and_stab_units_coord = data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord ;          
                good_and_stable_cl_depths = -[good_and_stab_units_coord.z];
                brain_region = {good_and_stab_units_coord.brain_region};                
            catch
                good_and_stable_cl_depths = 3840 - round(data.(subjects{i}).(sessions{j}).NPX_probes(p).templateDepths(good_and_stable_clusters+1));
                brain_region = [];
            end
            
            sp.clu_good = ( NaN(length(sp.clu), 1) );
            figure('units','normalized','outerposition',[0 0.1 1 0.85]);
            plotCount = 0;
            trGrnumb = length(unique(TrialGroups));
            for cl = 1:length(good_and_stable_clusters)
                st = sp.st(sp.clu==good_and_stable_clusters(cl));
                plotCount = plotCount+1;
                if plotCount>60
                    plotCount=1;
                figure('units','normalized','outerposition',[0 0.1 1 0.85]);
                end
                
                subplot(6, 10, plotCount)
                hold on
                for g=1:trGrnumb
                    trInd = find(TrialGroups==g);
                    fr = calcFR(st, EventTimes(trInd), binSize, PSTHwindow, sigma, [], [], 'trials');
                    [~, ~, fr_conf, ~] = normfit(fr);
                    plot(PSTHwindow(1)+binSize:binSize:PSTHwindow(2), mean(fr), 'color', colors(g,:))
                    ciplot(fr_conf(1,:), fr_conf(2,:), PSTHwindow(1)+binSize:binSize:PSTHwindow(2), colors(g,:), 0.25);
                end
                yl=ylim;
                plot([0 0], yl, '--k')
                axis([PSTHwindow(1) PSTHwindow(2) yl])
                if isempty(brain_region)
                    title(['cluster ' num2str(good_and_stable_clusters(cl)),' - Depth: ' num2str(good_and_stable_cl_depths(cl))]);
                else
                    title(['cluster ' num2str(good_and_stable_clusters(cl)),' - Depth: ' num2str(good_and_stable_cl_depths(cl)) ', ' brain_region{cl}]);
                end
            end
        end
 
    end

end

