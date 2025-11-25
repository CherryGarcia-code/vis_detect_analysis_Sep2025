function allUnitsSumm = calcTFTuning_main(data, subj_ind, allUnitsSumm)

tic;

probeNumbTot = 0;

subjects = fieldnames(data);
for i = subj_ind
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probesNumb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        probeNumbTot = probeNumbTot + probesNumb;
    end
end

preEventWind = 0.5;
postEventWind = 0.7;
PSTHwindow = [-preEventWind postEventWind];
binSize = 0.001;
sigma = 0.02;
% TFbins = [0.5 0.85 0.93 1.01 1.09 1.2 10];
% TFbins = [0.5 0.85 0.95 1.05 1.15 1.25 10];
TFbins = [0.5 0.79 0.85 0.9 0.94 0.98 1.03 1.08 1.13 1.2 1.3 10];

subjects = fieldnames(data);
probeCounter = 0;

if ~isfield(allUnitsSumm, 'TF')
    allUnitsSumm.TFparams.PSTHwindow = PSTHwindow;
    allUnitsSumm.TFparams.binSize = binSize;
    allUnitsSumm.TFparams.sigma = sigma;
    allUnitsSumm.TFparams.TFbins = TFbins;
    allUnitsSumm.TF = [];
end
 
allUnitsSubjInd = [allUnitsSumm.Units.subjInd];
allUnitsSesInd = [allUnitsSumm.Units.sesInd];
allUnitsProbeInd = [allUnitsSumm.Units.probeInd];
allUnitsCluInd = [allUnitsSumm.Units.cluInd];

for i = subj_ind
    
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probesNumb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        
        BaselineONTimes = data.(subjects{i}).(sessions{j}).NI_events.Baseline_ON.rise_t;
        ChangeONDur = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.duration;
        frameTimes = data.(subjects{i}).(sessions{j}).NI_events.frame_times_tr.time;
        TrialsData = data.(subjects{i}).(sessions{j}).behav_data.trials_data_exp;
        trialsNumb  = length(BaselineONTimes);
        hitTrials = ([TrialsData.IsHit]==1);
        missTrials = ([TrialsData.IsMiss]==1);
       
        for p = 1:probesNumb
            
            ind = find( allUnitsSubjInd==i&allUnitsSesInd==j&allUnitsProbeInd==p);
            clustersToUse = allUnitsCluInd(ind);
            
            probeCounter = probeCounter + 1;
            waitbar(probeCounter/probeNumbTot)
            
            sp = data.(subjects{i}).(sessions{j}).NPX_probes(p);
%             good_clusters = data.(subjects{i}).(sessions{j}).NPX_probes(p).cluster_id_good_and_stable;
            good_cl_coord = data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord;
            resultsSes = [];

            parfor cl = 1:length(clustersToUse)

                st = sp.st(sp.clu==clustersToUse(cl));

                TFtimesDecr = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(1), TFbins(2), 'lin', PSTHwindow(2), 1:trialsNumb);  % largest TF decrease
                TFtimesDecr2 = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(2), TFbins(3), 'lin', PSTHwindow(2), 1:trialsNumb); %2nd largest TF decrease
                TFtimesIncr = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(end-1), TFbins(end), 'lin', PSTHwindow(2), 1:trialsNumb); %largest TF increase
                TFtimesIncr2 = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(end-2), TFbins(end-1), 'lin', PSTHwindow(2), 1:trialsNumb); %2nd largest TF increase
                TFtimesBaseTF = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, -0.5, 0.5, 'log2', PSTHwindow(2), 1:trialsNumb); % Avg fr around 1Hz baseline TF, used to account for changes in mean fr vs time (ramps up/down)   
                
                frAvgTFdecr = calcFR(st,  cell2mat(TFtimesDecr(:)'), binSize, PSTHwindow, sigma, [], []);
                frAvgTFdecr2 = calcFR(st,  cell2mat(TFtimesDecr2(:)'), binSize, PSTHwindow, sigma, [], []);
                frAvgTFincr = calcFR(st,  cell2mat(TFtimesIncr(:)'), binSize, PSTHwindow, sigma, [], []);
                frAvgTFincr2 = calcFR(st,  cell2mat(TFtimesIncr2(:)'), binSize, PSTHwindow, sigma, [], []);    
                frAvgTFBase = calcFR(st,  cell2mat(TFtimesBaseTF(:)'), binSize, PSTHwindow, 0.05, [], []); % use larger smoothing here to reduce noise   
%                 frSdTFBase = std(frAvgTFBase(1:round(preEventWind/binSize)));

                [~, baseSD] = calcNormFr(calcFR(st,  cell2mat(TFtimesBaseTF(:)'), binSize, PSTHwindow, sigma, [], []), binSize, preEventWind, []);

                normFrTFmean = calcNormFr(frAvgTFincr+frAvgTFincr2, binSize, preEventWind, baseSD)-calcNormFr(frAvgTFdecr+frAvgTFdecr2, binSize, preEventWind, baseSD); 

                if abs(max(normFrTFmean))>abs(min(normFrTFmean))
                    indPeak = find(normFrTFmean==max(normFrTFmean),1,'first');
                else
                    indPeak = find(normFrTFmean==min(normFrTFmean),1,'first');
                end
               [indPeakStart, indPeakEnd] = calcPeakHWind(normFrTFmean, indPeak); 
               
                TFtimesDecrSeq = get_TF_pulses_seq(TrialsData, ChangeONDur, frameTimes,{[TFbins(1),TFbins(2)], [TFbins(1),TFbins(2)]}, 0.05, 'lin', PSTHwindow(2), 1:trialsNumb);
                TFtimesIncrSeq = get_TF_pulses_seq(TrialsData, ChangeONDur, frameTimes,{[TFbins(1),TFbins(2)], [TFbins(end-1),TFbins(end)]}, 0.05, 'lin', PSTHwindow(2), 1:trialsNumb);
                frAvgTFdecrSeq = calcFR(st,  cell2mat(TFtimesDecrSeq(:)'), binSize, PSTHwindow, sigma, [], []);
                frAvgTFincrSeq = calcFR(st,  cell2mat(TFtimesIncrSeq(:)'), binSize, PSTHwindow, sigma, [], []);
                normFrTFmean = calcNormFr(frAvgTFincrSeq, binSize, preEventWind, baseSD)-calcNormFr(frAvgTFdecrSeq, binSize, preEventWind, baseSD); 
                
                if abs(max(normFrTFmean))>abs(min(normFrTFmean))
                    indPeakSeq = find(normFrTFmean==max(normFrTFmean),1,'first');
                else
                    indPeakSeq = find(normFrTFmean==min(normFrTFmean),1,'first');
                end

               [indPeakSeqStart, indPeakSeqEnd] = calcPeakHWind(normFrTFmean, indPeakSeq);
               
                tuningCurveTF = zeros(1,length(TFbins)+1);     
                tuningCurveTFconf = zeros(length(TFbins)+1, 2);   
                TFbinsMean = zeros(1,length(TFbins)+1); 
                normFrTFcombTMP = [];
                anovaLabel = [];
                frTFbinAvg = zeros(length(TFbins)+1, (PSTHwindow(2)-PSTHwindow(1))/binSize);
                if ~isempty(indPeak)
                    for q = 1:length(TFbins)+1
                        
                        if q==1
                           [TFtimes, TFframeInd] = get_TF_pulses_seq(TrialsData, ChangeONDur, frameTimes,{[TFbins(1),TFbins(2)], [TFbins(1),TFbins(2)]}, 0.05, 'lin', postEventWind, 1:trialsNumb);
                        else
                           if q==length(TFbins)+1
                               [TFtimes, TFframeInd] = get_TF_pulses_seq(TrialsData, ChangeONDur, frameTimes,{[TFbins(end-1),TFbins(end)], [TFbins(end-1),TFbins(end)]}, 0.05, 'lin', postEventWind, 1:trialsNumb);
                           else
                               [TFtimes, TFframeInd] = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(q-1), TFbins(q), 'lin', postEventWind, 1:trialsNumb);
                           end
                        end

                        TFchange = mean(get_aligned_TF_from_frames_ind(TrialsData, TFframeInd, PSTHwindow)); 
                        indFrame = round(preEventWind/0.0167);
                        TFbinsMean(q) = mean(TFchange(indFrame+1:indFrame+3));  % 
                        if q==1
                            TFbinsMean(q) = TFbinsMean(q)*0.8;  %multipliers are just to separate sequences of TF pulses on TF axis 
                        else
                            if q==length(TFbins)+1
                                TFbinsMean(q) = TFbinsMean(q)*1.2;
                            end
                        end
                        
                        frTFbin = calcFR(st,  cell2mat(TFtimes(:)'), binSize, PSTHwindow, sigma, [], [], 'trials');
                        frTFbinAvg(q, :) = mean(frTFbin, 1);
                        
                        normFrTFmean = calcNormFr(frTFbin, binSize, preEventWind, baseSD)-calcNormFr(frAvgTFBase, binSize, preEventWind, baseSD); 
                        
                        if q==1||q==length(TFbins)+1
                            frChangeTFbin = mean(normFrTFmean(:,indPeakSeqStart:indPeakSeqEnd),2);       
                        else
                            frChangeTFbin = mean(normFrTFmean(:,indPeakStart:indPeakEnd),2);       
                        end
                        
                        tuningCurveTF(q) = mean(frChangeTFbin);
                        [~, ~, tuningCurveTFconf(q,:), ~] = normfit(frChangeTFbin);

                        if q>1&&q<length(TFbins)+1  %exclude TF sequences
                            normFrTFcombTMP = [normFrTFcombTMP ; frChangeTFbin];
                            anovaLabel = [anovaLabel repmat(q,1,length(frChangeTFbin))];
                        end
                    end
                    
                    pval = anovan(normFrTFcombTMP,{anovaLabel},'model',1,'display','off');

                
                    resultsSes(cl).brain_region = good_cl_coord(cl).brain_region_comb;
                    resultsSes(cl).tuningCurveTFpval = pval;
                    resultsSes(cl).tuningCurveTF = tuningCurveTF;
                    resultsSes(cl).tuningCurveTFconf = tuningCurveTFconf;
                    resultsSes(cl).TFbinsMean = TFbinsMean;
                    resultsSes(cl).TFbinAvgFr = frTFbinAvg;
                    resultsSes(cl).TFRespPeakTime = indPeak;
                    resultsSes(cl).TFRespPeakHW = indPeakEnd-indPeakStart;

                    
                else
                    resultsSes(cl).brain_region = good_cl_coord(cl).brain_region_comb;
                    resultsSes(cl).tuningCurveTFpval = NaN;
                    resultsSes(cl).tuningCurveTF = NaN(1,length(TFbins)+1);
                    resultsSes(cl).tuningCurveTFconf = NaN(length(TFbins)+1,2);
                    resultsSes(cl).TFbinsMean = NaN(1,length(TFbins)+1);    
                    resultsSes(cl).TFbinAvgFr = NaN(length(TFbins)+1, (PSTHwindow(2)-PSTHwindow(1))/binSize);
                    resultsSes(cl).TFRespPeakTime = NaN;
                    resultsSes(cl).TFRespPeakHW = NaN;
                end
                
            end
            
            allUnitsSumm.TF = [allUnitsSumm.TF resultsSes];
            
        end
    end
    
end

time = toc;
disp(['Finished ' subjects{i} ' in ' num2str(time/60) ' min'])


end

function [frTFnorm, sd] = calcNormFr(fr, binSize, preEventWind, sd)
     frBase = mean(fr(:,1:round((preEventWind/binSize))),1);   %avg across trials
     if isempty(sd)
         sd = std(frBase);
     end
     frTFnorm = (fr - (mean(frBase)))/sd; 
%      frTFnorm = (fr - (mean(frBase))); 
     frTFnorm(:, 1:round(preEventWind/binSize)) = [];  % leave only spikes after TF change onset
end


function [startInd, endInd] = calcPeakHWind(normFr, peakInd)
    peakFr = normFr(peakInd);
    if peakFr>0    % assumes the mean fr at baseline TF frames was subtracted 
        startInd = find(normFr(peakInd-1:-1:1)<=peakFr*0.5,1,'first');    %down from the peak value
        endInd = find(normFr(peakInd+1:end)<=peakFr*0.5,1,'first');   
    else
        startInd = find(normFr(peakInd-1:-1:1)>=peakFr*0.5,1,'first');    %down from the peak value
        endInd = find(normFr(peakInd+1:end)>=peakFr*0.5,1,'first');   
    end
    
    if isempty(startInd)
        startInd = 1;
    else
        startInd = peakInd - startInd;
    end
    if isempty(endInd)
        endInd = length(normFr);
    else
        endInd = peakInd + endInd;
    end
    
    if endInd-startInd<50 
        extra = floor((50-(endInd-startInd))/2);
        startInd = startInd - extra;
        endInd = endInd + extra;
        if startInd<1
            startInd = 1;
            endInd = 50;
        end
        if endInd>length(normFr)
            endInd=length(normFr);
            startInd = length(normFr) - 50;
        end
           
    end
    
end

