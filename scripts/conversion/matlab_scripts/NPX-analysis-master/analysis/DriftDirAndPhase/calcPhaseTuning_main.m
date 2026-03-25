function allUnitsSumm = calcPhaseTuning_main(data, subj_ind, allUnitsSumm)

probeNumbTot = 0;

subjects = fieldnames(data);
for i = subj_ind
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probesNumb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        probeNumbTot = probeNumbTot + probesNumb;
    end
end

% preEventWind = 1;
% postEventWind = 2;
% PSTHwindow = [-preEventWind postEventWind];

driftDir = [90 270];
phaseBins = 0:20:360;
phaseBinsPlot = phaseBins(1:end-1)+(phaseBins(2)-phaseBins(1))/2;
subjects = fieldnames(data);
probeCounter = 0;

if ~isfield(allUnitsSumm, 'Phase')  

    allUnitsSumm.PhaseParams.driftDir = driftDir;
    allUnitsSumm.PhaseParams.phaseBins = phaseBins;
    allUnitsSumm.PhaseParams.phaseBinsPlot = phaseBinsPlot;
    allUnitsSumm.Phase = [];
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
        ChangeONdur = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.duration;
        trialsNumb = length(TrialsData);
        
        for p = 1:probesNumb
            probeCounter = probeCounter + 1;
            waitbar(probeCounter/probeNumbTot)
            
            ind = find( allUnitsSubjInd==i&allUnitsSesInd==j&allUnitsProbeInd==p);
            clustersToUse = allUnitsCluInd(ind);
            
            sp = data.(subjects{i}).(sessions{j}).NPX_probes(p);
            good_cl_coord = data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord;
            [phaseBinStartTimesSes, phaseBinEndTimesSes] = getBinnedPhaseTimes(TrialsData, ChangeONdur, frameTimes, phaseBins, 1:trialsNumb, driftDir);
%             [phaseBinStartTimesSesShuffled, phaseBinEndTimesSesShuffled] = getBinnedPhaseTimes(TrialsData, ChangeONdur, frameTimes, phaseBins, randperm(trialsNumb,trialsNumb), driftDir);

            resultsSes = [];
            
            for cl = 1:length(clustersToUse)
                
                st = sp.st(sp.clu==clustersToUse(cl));
                phaseTuningCurveEvents = calcPhaseTuningCurve(st, phaseBinStartTimesSes, phaseBinEndTimesSes, 'events');
%                 phaseTuningCurveEventsShuffled = calcPhaseTuningCurve(st, phaseBinStartTimesSesShuffled, phaseBinEndTimesSesShuffled, 'events');
                pvalDriftDir = [];
                
                for q=1:size(phaseTuningCurveEvents,1)  % drift dir
                    anovaLabelPhaseBin = [];
                    for k=1:size(phaseTuningCurveEvents,2) % phase bins
%                         anovaLabelDriftDir = [anovaLabelDriftDir q*ones(1,length(phaseTuningCurveEvents{q,k}))];
%                         try
                        anovaLabelPhaseBin = [anovaLabelPhaseBin phaseBinsPlot(k)*ones(1,length(phaseTuningCurveEvents{q,k}))];
%                         catch
%                             anovaLabelPhaseBin = phaseBinsPlot(k)*ones(1,length(phaseTuningCurveEvents{q,k}));
%                         end
                    end
                    
                    pvalDriftDir(q) =  anovan(cell2mat(phaseTuningCurveEvents(g,:)), {anovaLabelPhaseBin},'model', 1, 'display', 'off');
                    
                end
                
      
%                pval =  anovan(cell2mat(phaseTuningCurveEvents(:)'), {anovaLabelDriftDir, anovaLabelPhaseBin},'model', 2);
%                figure
%                plot(cellfun(@mean, phaseTuningCurveEvents)')

                resultsSes(cl).brain_region = good_cl_coord(cl).brain_region_comb;
                resultsSes(cl).phaseTuningCurves = cellfun(@mean, phaseTuningCurveEvents)';
                resultsSes(cl).pvalDriftDir = pvalDriftDir;


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