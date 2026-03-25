function allUnitsSumm = calcChangeTuning_main(data, subj_ind, allUnitsSumm)

probeNumbTot = 0;

subjects = fieldnames(data);
for i = subj_ind
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probesNumb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        probeNumbTot = probeNumbTot + probesNumb;
    end
end

preEventWind = 1;
postEventWind = 2;
PSTHwindow = [-preEventWind postEventWind];
binSize = 0.001;
sigma = 0.02;
minRT = 0.6; 
maxRT = 2;
minRespDur = 0.2;
ChangeMagn = [1 1.25 1.35 1.5 2 4];

subjects = fieldnames(data);
probeCounter = 0;

pval = 0.01;
ZscoreThresh = -norminv(pval/(2*length(0:binSize:minRT)));

if ~isfield(allUnitsSumm, 'Change')  
    allUnitsSumm.ChangeParams.PSTHwindow = PSTHwindow;
    allUnitsSumm.ChangeParams.binSize = binSize;
    allUnitsSumm.ChangeParams.sigma = sigma;
    allUnitsSumm.ChangeParams.minRT = minRT;
    allUnitsSumm.ChangeParams.maxRT = maxRT;
    allUnitsSumm.ChangeParams.ChangeMagn = ChangeMagn;
    allUnitsSumm.ChangeParams.ZscoreThresh = ZscoreThresh;
    allUnitsSumm.Change = [];
end

allUnitsSubjInd = [allUnitsSumm.Units.subjInd];
allUnitsSesInd = [allUnitsSumm.Units.sesInd];
allUnitsProbeInd = [allUnitsSumm.Units.probeInd];
allUnitsCluInd = [allUnitsSumm.Units.cluInd];

for i = subj_ind
    
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probesNumb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        
        frameTimes = data.(subjects{i}).(sessions{j}).NI_events.frame_times_tr.time;
        TrialsData = data.(subjects{i}).(sessions{j}).behav_data.trials_data_exp;
        hitTrials = ([TrialsData.IsHit]==1);
        missTrials = ([TrialsData.IsMiss]==1);
        Change_magn = [TrialsData.Stim2TF];
        ChangeONtimes = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.rise_t;
        BaselineONtimes = data.(subjects{i}).(sessions{j}).NI_events.Baseline_ON.rise_t;
        
        ChangeONTimesNew = calcTrueChangeTimes(TrialsData, hitTrials, missTrials, frameTimes, ChangeONtimes, BaselineONtimes);
        
        if sum((ChangeONTimesNew-ChangeONtimes)>0.1)>0
            maxdelay = max(ChangeONTimesNew-ChangeONtimes);
            disp([sessions{j} '- found change delay up to ' num2str(1000*maxdelay) ' ms'])
        end

        Stim1Ori = [TrialsData.Stim1Ori];
        TF = {TrialsData.TF};

        ReactionTimes = [TrialsData.reactiontimes];
        ReactionTimesHits = [ReactionTimes.RT];
        ZscoreAvgWind = 1+round(preEventWind/binSize):round((preEventWind + minRT)/binSize);
        
        for p = 1:probesNumb
            probeCounter = probeCounter + 1;
            waitbar(probeCounter/probeNumbTot)
            
            ind = find( allUnitsSubjInd==i&allUnitsSesInd==j&allUnitsProbeInd==p);
            clustersToUse = allUnitsCluInd(ind);
            
            sp = data.(subjects{i}).(sessions{j}).NPX_probes(p);
            good_cl_coord = data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord;
            
            resultsSes = [];
            
            parfor cl = 1:length(clustersToUse)
                tuningCurveChangeHits = zeros(1,length(ChangeMagn));     
                tuningCurveChangeMisses = zeros(1,length(ChangeMagn));     
                ZscCombHits = [];
                anovaLabel = [];
                
                st = sp.st(sp.clu==clustersToUse(cl));
                TrialsToUse = find( ((hitTrials==1)&(ReactionTimesHits>minRT)&(ReactionTimesHits<maxRT))|(missTrials==1));
                frAvgtmp = calcFR(st, ChangeONTimesNew(TrialsToUse), binSize, PSTHwindow, sigma, [], []);
                sdBase = std(frAvgtmp(1:round(-PSTHwindow(1)/binSize)));
                
                TrialsToUse = find( ((hitTrials==1)&(ReactionTimesHits>minRT)&(ReactionTimesHits<maxRT)));
                frAvgtmp = calcFR(st, ChangeONTimesNew(TrialsToUse), binSize, PSTHwindow, sigma, [], []);
                ZscoreHitsAvg = calcZscore(frAvgtmp, binSize, preEventWind, sdBase);
                
                if sum(abs(ZscoreHitsAvg(ZscoreAvgWind))>ZscoreThresh)>=(1000*minRespDur)  % at least xxxms of activity above sign threshold
                    isChangeResp = 1;
                else
                    isChangeResp = 0;
                end
%                 
%                 ZscoreHits = zeros(length(ChangeMagn), (PSTHwindow(2)-PSTHwindow(1))/binSize);
%                 ZscoreMisses = zeros(length(ChangeMagn), (PSTHwindow(2)-PSTHwindow(1))/binSize);
%                 
                frHits = zeros(length(ChangeMagn), (PSTHwindow(2)-PSTHwindow(1))/binSize);
                frMisses = zeros(length(ChangeMagn), (PSTHwindow(2)-PSTHwindow(1))/binSize);
                
                for q = 1:length(ChangeMagn)
                    hitTrialsToUse = find(hitTrials==1&Change_magn==ChangeMagn(q)&(ReactionTimesHits>minRT)&(ReactionTimesHits<maxRT));
                    missTrialsToUse = find(missTrials==1&Change_magn==ChangeMagn(q));

%                     frHits_tr = calcFR(st, ChangeONTimesNew(hitTrialsToUse), binSize, PSTHwindow, sigma, [], [], 'trials');
                    frMisses(q, :) = calcFR(st, ChangeONTimesNew(missTrialsToUse), binSize, PSTHwindow, sigma, [], []);
                    frHits(q, :) = calcFR(st, ChangeONTimesNew(hitTrialsToUse), binSize, PSTHwindow, sigma, [], []);
                    
%                     ZscoreHitsChangeMagn = calcZscore(frHits_tr, binSize, preEventWind, sdBase);
%                     ZscoreAvgWindHitsChangeMagn = nanmean(ZscoreHitsChangeMagn(:, ZscoreAvgWind), 2); % average Zscore from Change onset to min RT 
                    
%                     ZscoreHits(q, :) = nanmean(ZscoreHitsChangeMagn, 1);  % average across trials
%                     tuningCurveChangeHits(q) = nanmean(ZscoreAvgWindHitsChangeMagn);
                                            
%                     ZscoreMisses(q, :) = calcZscore(frMisses, binSize, preEventWind, sdBase);
%                     tuningCurveChangeMisses(q) = nanmean(ZscoreMisses(q, ZscoreAvgWind));
                    
%                     ZscCombHits = [ZscCombHits ; ZscoreAvgWindHitsChangeMagn];
%                     anovaLabel = [anovaLabel repmat(ChangeMagn(q),1,length(ZscoreAvgWindHitsChangeMagn))];
                end 
                
%                 try
%                     pvalHits = anovan(ZscCombHits, {anovaLabel},'model',1,'display','off');
%                 catch
%                     pvalHits = NaN;
%                 end

                resultsSes(cl).brain_region = good_cl_coord(cl).brain_region_comb;
                resultsSes(cl).isChangeResp = isChangeResp;
                resultsSes(cl).ZscoreHitsAvg = ZscoreHitsAvg;
%                 resultsSes(cl).tuningCurveChangeHitspval = pvalHits;
%                 resultsSes(cl).tuningCurveChangeHits = tuningCurveChangeHits;
%                 resultsSes(cl).tuningCurveChangeMisses = tuningCurveChangeMisses;
%                 resultsSes(cl).ZscoreHits = ZscoreHits;
%                 resultsSes(cl).ZscoreMisses = ZscoreMisses;
                resultsSes(cl).frHits = frHits;
                resultsSes(cl).frMisses = frMisses;

            end
            
            allUnitsSumm.Change = [allUnitsSumm.Change resultsSes];
            
        end
    end
    
end

disp(['Finished ' subjects{i} ])


end



function [Zscore, sd] = calcZscore(fr, binSize, preEventWind, sd)
     frBase = nanmean(fr(:,1:round(preEventWind/binSize)),1); 
     if isempty(sd)
         sd = nanstd(frBase);
     end
     Zscore = (fr - (nanmean(frBase)))/sd; 
end