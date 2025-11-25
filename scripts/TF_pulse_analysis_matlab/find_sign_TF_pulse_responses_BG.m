clearvars -except data

% pre_event_wind = 0.6;
% post_event_wind = 1.3;
 pre_event_wind = .5;
 post_event_wind = 1;
sp_count_wind = 0.5;
PSTHwindow = [-pre_event_wind post_event_wind];
binSize = 0.001;

%load('/mnt/mlohse/winstor/swc/mrsic_flogel/public/projects/AnKh_20200820_NPX_DMDM/GLMOutputs/AllUnitsGLMSumm_v6.mat')
%p_val=AllUnitsGLMSumm.TFResponsiveIdx;
%p_val(isnan(p_val))=0;
subjects = fieldnames(data);

sign_cl = [];
cl_counter = 0;
clall=0;
SessCount=0;
for i = 1:length(subjects)
    clear sessions Baseline_ON_times Change_ON_times Change_ON_dur frame_times TrialsData trials_numb
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        clear Baseline_ON_times Change_ON_times Change_ON_dur frame_times TrialsData trials_numb TF_incr_frame_times TF_decr_frame_times EventTimes TrialGroups
        SessCount=SessCount+1;
        probes_numb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        
        Baseline_ON_times = data.(subjects{i}).(sessions{j}).NI_events.Air_puff.rise_t;
        AirpuffTimes = data.(subjects{i}).(sessions{j}).NI_events.Baseline_ON.rise_t;
        RewTimes = data.(subjects{i}).(sessions{j}).NI_events.Baseline_ON.rise_t;
        
        Change_ON_times = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.rise_t;
        Change_ON_dur = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.duration;
        frame_times = data.(subjects{i}).(sessions{j}).NI_events.frame_times_tr.time;
        TrialsData = data.(subjects{i}).(sessions{j}).behav_data.trials_data_exp;
        trials_numb  = length(Baseline_ON_times);
        
        %MotionOnsets=data.(subjects{i}).(sessions{j}).Video.MotionOnsetTimes;


        for t=1:length(TrialsData)
            if TrialsData(t).IsFA
                MotionOnsets(t)=Baseline_ON_times(t)+TrialsData(t).reactiontimes.FA;
                 elseif TrialsData(t).IsAbort
                 MotionOnsets(t)=Baseline_ON_times(t)+TrialsData(t).reactiontimes.abort;
            else
                MotionOnsets(t)=NaN;

            end
        end
        
        TF_incr_frame_times = get_TF_pulses_v2_2023(TrialsData, Change_ON_dur, frame_times, MotionOnsets,  1, 20, 'log2', PSTHwindow(2), 1:trials_numb);  % largest TF decrease
        TF_decr_frame_times = get_TF_pulses_v2_2023(TrialsData, Change_ON_dur, frame_times, MotionOnsets, -20, -1, 'log2', PSTHwindow(2), 1:trials_numb);  % largest TF decrease
        
        EventTimes = [cell2mat(TF_incr_frame_times(:)')  cell2mat(TF_decr_frame_times(:)')];
        TrialGroups = ones(1, length(EventTimes));
        TrialGroups(1+length(cell2mat(TF_incr_frame_times(:)')):end) = 2;   % decreases in TF
        
        for p = 1:probes_numb
            clear sp good_clusters
            sp = data.(subjects{i}).(sessions{j}).NPX_probes(p);
            good_clusters = data.(subjects{i}).(sessions{j}).NPX_probes(p).cluster_id_KS_good;
            for cl = 1:length(good_clusters) %[401:499 738]
                clear st spikes_TF_up spikes_TF_down
                
                clall=clall+1;
               % disp(clall)
                
%                if p_val(clall) > 0.01
%                    keyboard
%                end
%                     
%                     
%                     if startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'MOs')
%                         RegionId=1;
%                     elseif contains(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'CP')
%                         RegionId=2;
%                     elseif contains(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'VISp')
%                         RegionId=3;
%                     elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'SIM')
%                         RegionId=4;
%                     elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'SC')
%                         RegionId=5;
%                     elseif contains(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'LP')
%                         RegionId=6;
%                     elseif contains(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'MRN')
%                         RegionId=7;
%                     elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'ACA')
%                         RegionId=8;
%                     elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'CA1')
%                         RegionId=9;
%                     elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'PL')
%                         RegionId=10;
%                     elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'VISp')
%                         RegionId=11;
%                     else
%                         RegionId=0;
%                     end
                    
                    %                                      if RegionId ~= 1
                    %                                          continue
                    %                                      end
                    
                    st = sp.st(sp.clu==good_clusters(cl));
                    [~, ~, ~, ~, ~, spikes_TF_up] = psthAndBA(st, EventTimes(TrialGroups==1), PSTHwindow, binSize);
                    [~, ~, ~, ~, ~, spikes_TF_down] = psthAndBA(st, EventTimes(TrialGroups==2), PSTHwindow, binSize);
                    
                     [~, p_val(clall)] = ttest2(mean(spikes_TF_up(:, 1+1000*pre_event_wind:1000*(pre_event_wind+sp_count_wind)),2), mean(spikes_TF_down(:, 1+1000*pre_event_wind:1000*(pre_event_wind+sp_count_wind)),2) );
                    %      [~, p_val(clall)] = ttest2(mean(spikes_TF_up(:, 1+1000*pre_event_wind:1000*(pre_event_wind+sp_count_wind)),2), mean(spikes_TF_down(:, 1+1000*pre_event_wind:1000*(pre_event_wind+sp_count_wind)),2) );
                    % p_val(clall)=NaN;
                        
                        cl_counter = cl_counter+1;
                        disp(cl_counter)
                        sign_cl.ind(cl_counter, :) = [i; j; p; good_clusters(cl); p_val(clall);cl;cl_counter;SessCount];
                     %   sign_cl.RegionId{cl_counter}=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region;
                        
                     %   sign_cl.x(cl_counter)=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).x;
                     %   sign_cl.y(cl_counter)=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).y;
                     %   sign_cl.z(cl_counter)=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).z;
                        
                        
                        %                      mean(conv2(smWin, 1, spikes_TF_up', 'valid')'./binSize);
                        
                        sign_cl.spikes_TF_up(:,cl_counter) = mean(smoothdata(spikes_TF_up','gaussian',50)'); % normally this is 50
                        sign_cl.spikes_TF_down(:,cl_counter) = mean(smoothdata(spikes_TF_down','gaussian',50)');
                        sign_cl.spikes_TF_upSEM(:,cl_counter) = std(smoothdata(spikes_TF_up','gaussian',50)')./sqrt(size(spikes_TF_up,1));
                        sign_cl.spikes_TF_downSEM(:,cl_counter) = std(smoothdata(spikes_TF_down','gaussian',50)')./sqrt(size(spikes_TF_down,1));
                        
                     %   sign_cl.Region(cl_counter)=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl);
                    
             %  end
            end
        end
    end
end

% cd('/mnt/data/mlohse/')
%save('PopTFs_All_thresh1std_300123', 'sign_cl', 'p_val','clall', 'subjects','-v7.3')
%save('PopTFs_All_thresh1std_300323_40msHW', 'sign_cl', 'p_val','clall', 'subjects','-v7.3')
% save('PopTFs_All_thresh1std_280323', 'sign_cl', 'p_val','clall', 'subjects','-v7.3')

% 
% figure;
% subplot(1,2,1)
% shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(sign_cl.spikes_TF_up,'gaussian',200),2),std(smoothdata(sign_cl.spikes_TF_up,'gaussian',200)')./sqrt(size(sign_cl.spikes_TF_up,2)),'lineProps',{'linewidth',2})
% hold on
% shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(sign_cl.spikes_TF_down,'gaussian',200),2),std(smoothdata(sign_cl.spikes_TF_down,'gaussian',200)')./sqrt(size(sign_cl.spikes_TF_up,2)),'lineProps',{'linewidth',2})
% subplot(1,2,2)
% plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(sign_cl.spikes_TF_up,'gaussian',200),2),'linewidth',2)
% hold on
% plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(sign_cl.spikes_TF_down,'gaussian',200),2),'linewidth',2)
% 
% UsedSess=unique(sign_cl.ind(:, 9))
% for s=1:length(unique(sign_cl.ind(:, 9)))
%     SessMeanUp(:,s)=mean(smoothdata(sign_cl.spikes_TF_up(:,find(sign_cl.ind(:, 9)==UsedSess(s))),'gaussian',200),2)
%     SessMeanDown(:,s)=mean(smoothdata(sign_cl.spikes_TF_down(:,find(sign_cl.ind(:, 9)==UsedSess(s))),'gaussian',200),2)
% end
% prePulse=mean(SessMeanUp(101:400,:));
% prePulseSTD=std(SessMeanUp(101:400,:));
% 
% SessMeanUpNorm=(SessMeanUp-prePulse)./prePulseSTD
% SessMeanDownNorm=(SessMeanDown-prePulse)./prePulseSTD
% 
% for s=1:length(unique(sign_cl.ind(:, 9)))
%     len(s)=length(find(sign_cl.ind(:, 9)==UsedSess(s)));
% end
% 
% decentUnitNo=find(len>5)
% 
% figure;
% subplot(1,2,1)
% shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(SessMeanUp(:,decentUnitNo),2),std(SessMeanUp(:,decentUnitNo)')./sqrt(size(SessMeanUp(:,decentUnitNo),2)),'lineProps',{'linewidth',2})
% hold on
% shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(SessMeanDown(:,decentUnitNo),2),std(SessMeanDown(:,decentUnitNo)')./sqrt(size(SessMeanDown(:,decentUnitNo),2)),'lineProps',{'linewidth',2})
% subplot(1,2,2)
% plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanUp(:,decentUnitNo),'linewidth',2)
% hold on
% plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanDown(:,decentUnitNo),'linewidth',2)
% 
% figure(10101);
% subplot(1,2,1)
% shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(SessMeanUpNorm(:,decentUnitNo),2),std(SessMeanUpNorm(:,decentUnitNo)')./sqrt(size(SessMeanUpNorm(:,decentUnitNo),2)),'lineProps',{'linewidth',2,'color',[0 114 178]/255})
% hold on
% shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(SessMeanDownNorm(:,decentUnitNo),2),std(SessMeanDownNorm(:,decentUnitNo)')./sqrt(size(SessMeanDownNorm(:,decentUnitNo),2)),'lineProps',{'linewidth',2,'color',[213 94 0]/255})
% ylabel('Z-score (sessions)')
% xlabel('Time from pulse (sec)')
% 
% subplot(1,2,2)
% plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanUpNorm(:,decentUnitNo),'linewidth',2)
% hold on
% plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanDownNorm(:,decentUnitNo),'linewidth',2)
% ylabel('Z-score (sessions)')
% xlabel('Time from pulse (Sec)')
% set(findall(gcf,'-property','FontSize'),'FontSize',12)
% 
% 
% 
% figure(202);
% for s=1:30
%     subplot(6,5,s)
%     plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanUpNorm(:,decentUnitNo(s)),'linewidth',2)
%     hold on
%     plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanDownNorm(:,decentUnitNo(s)),'linewidth',2)
%     ylabel('Z-score (sessions)')
%     xlabel('Time from pulse (Ssense ec)')
%     % set(findall(gcf,'-property','FontSize'),'FontSize',12)
% end
% 
% 
% 
