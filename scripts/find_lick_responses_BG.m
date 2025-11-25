clearvars -except data AllUnitsGLMSumm

pre_event_wind = 2;
post_event_wind = .75;
sp_count_wind = 0.5;
PSTHwindow = [-pre_event_wind post_event_wind];
binSize = 0.001;

subjects = fieldnames(data);
lick_cl = [];
cl_counter = 0;
clall=0;
SessCount=0;
for i = 1:length(subjects)
    clear sessions Baseline_ON_times Change_ON_times Change_ON_dur frame_times TrialsData trials_numb
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        disp(sessions{j})
        clear Baseline_ON_times Change_ON_times Change_ON_dur frame_times TrialsData trials_numb TF_incr_frame_times TF_decr_frame_times EventTimes TrialGroups
        SessCount=SessCount+1;
        probes_numb = length(data.(subjects{i}).(sessions{j}).NPX_probes);

        Baseline_ON_times = data.(subjects{i}).(sessions{j}).NI_events.Baseline_ON.rise_t;
        Change_ON_times = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.rise_t;
        Change_ON_dur = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.duration;
        frame_times = data.(subjects{i}).(sessions{j}).NI_events.frame_times_tr.time;
        TrialsData = data.(subjects{i}).(sessions{j}).behav_data.trials_data_exp;
        trials_numb  = length(Baseline_ON_times);

        %         if isempty(data.(subjects{i}).(sessions{j}).Video)


        for t=1:length(TrialsData)
            if TrialsData(t).IsFA
                lick_ON_times(t)=Baseline_ON_times(t)+TrialsData(t).reactiontimes.FA;
                % elseif TrialsData(t).IsAbort
                % lick_ON_times(t)=Baseline_ON_times(t)+TrialsData(t).reactiontimes.abort;
                % elseif TrialsData(t).IsHit
                % lick_ON_times(t)=Baseline_ON_times(t)+TrialsData(t).reactiontimes.RT+TrialsData(t).stimT;
                % elseif TrialsData(t).IsMiss
                % lick_ON_times(t)=Baseline_ON_times(t)+TrialsData(t).reactiontimes.Miss+TrialsData(t).stimT;
                % elseif TrialsData(t).IsMiss
            else
                lick_ON_times(t)=NaN;

            end
        end





        %   lick_ON_times = data.(subjects{i}).(sessions{j}).NI_events.Air_puff.rise_t-0.2; % based on airpuff with 200 ms substracted as delay between piezo detection and guessed motion onset (basd on mean diference between motion onset and piezo in other sessions)
        %         else
        %   MotionOnsets=data.(subjects{i}).(sessions{j}).Video.MotionOnsetTimes;
        % lick_ON_times = MotionOnsets;
        %   lick_ON_times(([TrialsData.IsAbortWithFA]+[TrialsData.IsFA])==0)=NaN;   % find all early licks
        %         end




        LickDelay=lick_ON_times-Baseline_ON_times;
        lick_ON_times(LickDelay<3)=NaN;

        EventTimes = lick_ON_times;
        TrialGroups = ones(1, length(EventTimes));
        TrialGroups(isnan(EventTimes))=0;

        for p = 1:probes_numb
            clear sp good_clusters
            sp = data.(subjects{i}).(sessions{j}).NPX_probes(p);
            good_clusters = data.(subjects{i}).(sessions{j}).NPX_probes(p).cluster_id_KS_good;
            for cl = 1:length(good_clusters) %[401:499 738]
                clear st spikes_TF_up spikes_TF_down


                %                 if (startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'MOs') & data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).y<1.5)
                %                     RegionId=1;
                %                 elseif contains(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'CP')
                %                     RegionId=2;
                %                 elseif contains(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'VISp')
                %                     RegionId=3;
                %                 elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'SIM')
                %                     RegionId=4;
                %                 elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'SC')
                %                     RegionId=5;
                %                 elseif contains(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'LP')
                %                     RegionId=6;
                %                 elseif contains(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'MRN')
                %                     RegionId=7;
                %                 elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'ACA')
                %                     RegionId=8;
                %                 elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'CA1')
                %                     RegionId=9;
                %                 elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'PL')
                %                     RegionId=10;
                %                 elseif startsWith(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'VISp')
                %                     RegionId=11;
                %                 elseif strcmp(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'V')
                %                     RegionId=12;
                %                 elseif strcmp(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'GPe')
                %                     RegionId=13;
                %                 elseif strcmp(data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region,'SNr')
                %                     RegionId=14;
                %                 else
                %                     RegionId=0;
                %                 end
                %
                %                 if RegionId ~= 1
                %                     continue
                %                 end

                clall=clall+1;
                st = sp.st(sp.clu==good_clusters(cl));
                [~, ~, ~, ~, ~, spikes_lick] = psthAndBA(st, EventTimes(TrialGroups==1), PSTHwindow, binSize);

                % [~, p_val(clall)] = ttest2(mean(spikes_TF_up(:, 1+1000*pre_event_wind:1000*(pre_event_wind+sp_count_wind)),2), mean(spikes_TF_down(:, 1+1000*pre_event_wind:1000*(pre_event_wind+sp_count_wind)),2) );
                 
             
                %if p_val(clall)
                cl_counter = cl_counter+1;

                   offsets=[.2 .4];
                [~, p_val(cl_counter)] = ttest2(mean(spikes_lick(:, 1+1000*pre_event_wind-offsets(2)*1000:1000*(pre_event_wind)-offsets(1)*1000),2), mean(spikes_lick(:, 1:1000*(.25)),2) );

                lick_cl.ind(cl_counter, :) = [i; j; p; good_clusters(cl);cl;cl_counter;SessCount,p_val(cl_counter)];
                %   lick_cl.RegionId{cl_counter}=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).brain_region;

                % lick_cl.x(cl_counter)=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).x;
                % lick_cl.y(cl_counter)=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).y;
                % lick_cl.z(cl_counter)=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl).z;
                %                      mean(conv2(smWin, 1, spikes_TF_up', 'valid')'./binSize);
                lick_cl.spikes_lick(:,cl_counter) = mean(smoothdata(spikes_lick','gaussian',50)');
                lick_cl.spikes_lick_SEM(:,cl_counter) = std(smoothdata(spikes_lick','gaussian',50)')./sqrt(size(spikes_lick,1));

                % lick_cl.Region(cl_counter)=data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord(cl);
                %end
            end
        end
    end

end
%cd('/mnt/data/mlohse/')
%save('PoplickAll_motionOnsetWholeBrain301122', 'lick_cl', 'p_val','clall', 'subjects','-v7.3')
%save('PoplickAll_motionOnsetWholeBrain310123', 'lick_cl', 'p_val','clall', 'subjects','-v7.3')

figure
i=0;
for n=1:100
    i=i+1;
    subplot(10,10,i)
shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],lick_cl.spikes_lick(:,n),lick_cl.spikes_lick_SEM(:,n),'lineprops',{'linewidth',2,'color','b'})
end

figure;
shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(lick_cl.spikes_lick,'gaussian',200),2),std(smoothdata(lick_cl.spikes_lick,'gaussian',200)')./sqrt(size(lick_cl.spikes_lick,2)),'lineProps',{'linewidth',2})


figure;
subplot(1,2,1)
shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(lick_cl.spikes_TF_up,'gaussian',200),2),std(smoothdata(lick_cl.spikes_TF_up,'gaussian',200)')./sqrt(size(sign_cl.spikes_TF_up,2)),'lineProps',{'linewidth',2})
hold on
shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(lick_cl.spikes_TF_down,'gaussian',200),2),std(smoothdata(lick_cl.spikes_TF_down,'gaussian',200)')./sqrt(size(sign_cl.spikes_TF_up,2)),'lineProps',{'linewidth',2})
subplot(1,2,2)
plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(lick_cl.spikes_TF_up,'gaussian',200),2),'linewidth',2)
hold on
plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(smoothdata(lick_cl.spikes_TF_down,'gaussian',200),2),'linewidth',2)

UsedSess=unique(lick_cl.ind(:, 9))
for s=1:length(unique(sign_cl.ind(:, 9)))
    SessMeanUp(:,s)=mean(smoothdata(lick_cl.spikes_TF_up(:,find(lick_cl.ind(:, 9)==UsedSess(s))),'gaussian',200),2)
    SessMeanDown(:,s)=mean(smoothdata(lick_cl.spikes_TF_down(:,find(lick_cl.ind(:, 9)==UsedSess(s))),'gaussian',200),2)
end
prePulse=mean(SessMeanUp(101:400,:));
prePulseSTD=std(SessMeanUp(101:400,:));

SessMeanUpNorm=(SessMeanUp-prePulse)./prePulseSTD
SessMeanDownNorm=(SessMeanDown-prePulse)./prePulseSTD

for s=1:length(unique(sign_cl.ind(:, 9)))
    len(s)=length(find(sign_cl.ind(:, 9)==UsedSess(s)));
end

decentUnitNo=find(len>5)

figure;
subplot(1,2,1)
shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(SessMeanUp(:,decentUnitNo),2),std(SessMeanUp(:,decentUnitNo)')./sqrt(size(SessMeanUp(:,decentUnitNo),2)),'lineProps',{'linewidth',2})
hold on
shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(SessMeanDown(:,decentUnitNo),2),std(SessMeanDown(:,decentUnitNo)')./sqrt(size(SessMeanDown(:,decentUnitNo),2)),'lineProps',{'linewidth',2})
subplot(1,2,2)
plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanUp(:,decentUnitNo),'linewidth',2)
hold on
plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanDown(:,decentUnitNo),'linewidth',2)

figure(10101);
subplot(1,2,1)
shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(SessMeanUpNorm(:,decentUnitNo),2),std(SessMeanUpNorm(:,decentUnitNo)')./sqrt(size(SessMeanUpNorm(:,decentUnitNo),2)),'lineProps',{'linewidth',2,'color',[0 114 178]/255})
hold on
shadedErrorBar([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],mean(SessMeanDownNorm(:,decentUnitNo),2),std(SessMeanDownNorm(:,decentUnitNo)')./sqrt(size(SessMeanDownNorm(:,decentUnitNo),2)),'lineProps',{'linewidth',2,'color',[213 94 0]/255})
ylabel('Z-score (sessions)')
xlabel('Time from pulse (sec)')

subplot(1,2,2)
plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanUpNorm(:,decentUnitNo),'linewidth',2)
hold on
plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanDownNorm(:,decentUnitNo),'linewidth',2)
ylabel('Z-score (sessions)')
xlabel('Time from pulse (Sec)')
set(findall(gcf,'-property','FontSize'),'FontSize',12)



figure(202);
for s=1:30
    subplot(6,5,s)
    plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanUpNorm(:,decentUnitNo(s)),'linewidth',2)
    hold on
    plot([PSTHwindow(1)+binSize:binSize:PSTHwindow(2)],SessMeanDownNorm(:,decentUnitNo(s)),'linewidth',2)
    ylabel('Z-score (sessions)')
    xlabel('Time from pulse (Ssense ec)')
    % set(findall(gcf,'-property','FontSize'),'FontSize',12)
end

