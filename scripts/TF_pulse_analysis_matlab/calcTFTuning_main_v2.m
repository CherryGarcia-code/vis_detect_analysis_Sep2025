function allUnitsSumm = calcTFTuning_main_v2(data, subj_ind, allUnitsSumm, TFdistr)

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

% TFbins = makeTFbins(TFdistr, 0.08, 0.6, 1.5);
% TFbins = makeTFbins(TFdistr, 0.1, 0.5, 2);
TFbins = makeTFbins(TFdistr, 0.166, 0.5, 2);

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
% tuningCurveTFpval = [allUnitsSumm.TF.tuningCurveTFpval];

% fields = {'tuningCurveTF', 'tuningCurveTFconf', 'TFbinsMean', 'TFbinAvgFr'};
% fields = {'tuningCurveTFUp', 'tuningCurveDown', 'TFbinsMeanDrift', 'TFbinAvgFr'};

for i = subj_ind
    
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probesNumb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        
        BaselineONTimes = data.(subjects{i}).(sessions{j}).NI_events.Baseline_ON.rise_t;
        ChangeONDur = data.(subjects{i}).(sessions{j}).NI_events.Change_ON.duration;
        frameTimes = data.(subjects{i}).(sessions{j}).NI_events.frame_times_tr.time;
        TrialsData = data.(subjects{i}).(sessions{j}).behav_data.trials_data_exp;
        RewardTimes = data.(subjects{i}).(sessions{j}).NI_events.Valve_L.rise_t;
        trialsNumb = length(BaselineONTimes);
        
        ChangeONDur = RewardTimes-BaselineONTimes;  % for control mice, to remove TFs after valve opens
        ChangeONDur(isnan(ChangeONDur)) = 0;    
        for p = 1:probesNumb
            
            ind = find( allUnitsSubjInd==i&allUnitsSesInd==j&allUnitsProbeInd==p);
%             ind = find( allUnitsSubjInd==i&allUnitsSesInd==j&allUnitsProbeInd==p&tuningCurveTFpval<0.01);

            clustersToUse = allUnitsCluInd(ind);
%             TFpval = tuningCurveTFpval(ind);
            
            probeCounter = probeCounter + 1;
            waitbar(probeCounter/probeNumbTot)
            
            sp = data.(subjects{i}).(sessions{j}).NPX_probes(p);
            good_cl_coord = data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord;
            TFtimesDecr = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(1,1), TFbins(1,2), 'lin', PSTHwindow(2), 1:trialsNumb);  % largest TF decrease
%             TFtimesDecr2 = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(2,1), TFbins(2,2), 'lin', PSTHwindow(2), 1:trialsNumb); %2nd largest TF decrease
            TFtimesIncr = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(end,1), TFbins(end,2), 'lin', PSTHwindow(2), 1:trialsNumb); %largest TF increase
%             TFtimesIncr2 = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(end-1,1), TFbins(end-1,2), 'lin', PSTHwindow(2), 1:trialsNumb); %2nd largest TF increase
            TFtimesBaseTF = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, -0.5, 0.5, 'log2', PSTHwindow(2), 1:trialsNumb); % Avg fr around 1Hz baseline TF, used to account for changes in mean fr vs time (ramps up/down)   

            resultsSes = [];

            parfor cl = 1:length(clustersToUse)
%                 if TFpval(cl)<0.01
                    
                    st = sp.st(sp.clu==clustersToUse(cl));

                    frAvgTFdecr = calcFR(st,  cell2mat(TFtimesDecr(:)'), binSize, PSTHwindow, sigma, [], []);
%                     frAvgTFdecr2 = calcFR(st,  cell2mat(TFtimesDecr2(:)'), binSize, PSTHwindow, sigma, [], []);
                    frAvgTFincr = calcFR(st,  cell2mat(TFtimesIncr(:)'), binSize, PSTHwindow, sigma, [], []);
%                     frAvgTFincr2 = calcFR(st,  cell2mat(TFtimesIncr2(:)'), binSize, PSTHwindow, sigma, [], []);    
                    frAvgTFBase = calcFR(st,  cell2mat(TFtimesBaseTF(:)'), binSize, PSTHwindow, 0.1, [], []); % use larger smoothing here to reduce noise   
    %                 frSdTFBase = std(frAvgTFBase(1:round(preEventWind/binSize)));

                    [~, baseSD] = calcZscore(calcFR(st,  cell2mat(TFtimesBaseTF(:)'), binSize, PSTHwindow, sigma, [], []), binSize, preEventWind, []);

%                     normFrTFmean = calcZscore(frAvgTFincr+frAvgTFincr2, binSize, preEventWind, baseSD)-calcZscore(frAvgTFdecr+frAvgTFdecr2, binSize, preEventWind, baseSD); 
                    normFrTFmean = calcZscore(frAvgTFincr, binSize, preEventWind, baseSD)-calcZscore(frAvgTFdecr, binSize, preEventWind, baseSD); 

                    if abs(max(normFrTFmean))>abs(min(normFrTFmean))
                        indPeak = find(normFrTFmean==max(normFrTFmean),1,'first');
                    else
                        indPeak = find(normFrTFmean==min(normFrTFmean),1,'first');
                    end
                   [indPeakStart, indPeakEnd] = calcPeakHWind(normFrTFmean, indPeak); 

                    tuningCurveTF = zeros(1,size(TFbins,1));     
                    tuningCurveTFconf = zeros(size(TFbins,1), 2);   
                    TFbinsMean = zeros(1,size(TFbins,1)); 
                    normFrTFcombTMP = [];
                    anovaLabel = [];
                    frTFbinAvg = zeros(size(TFbins,1), (PSTHwindow(2)-PSTHwindow(1))/binSize);
                    if ~isempty(indPeak)
                        for q = 1:size(TFbins,1)

                           [TFtimes, TFframeInd] = get_TF_pulses_v2(TrialsData, ChangeONDur, frameTimes, TFbins(q,1), TFbins(q,2), 'lin', postEventWind, 1:trialsNumb);

                            TFchange = mean(get_aligned_TF_from_frames_ind(TrialsData, TFframeInd, 1:trialsNumb, PSTHwindow)); 
                            indFrame = round(preEventWind/0.0167);
                            TFbinsMean(q) = mean(TFchange(indFrame+1:indFrame+3));  

                            frTFbin = calcFR(st,  cell2mat(TFtimes(:)'), binSize, PSTHwindow, sigma, [], [], 'trials');
                            frTFbinAvg(q, :) = calcZscore(mean(frTFbin, 1), binSize, preEventWind, baseSD, 'keep_baseline');

                            normFrTFmean = calcZscore(frTFbin, binSize, preEventWind, baseSD)-calcZscore(frAvgTFBase, binSize, preEventWind, baseSD); 
                            frChangeTFbin = mean(normFrTFmean(:,indPeakStart:indPeakEnd),2);       

                            tuningCurveTF(q) = mean(frChangeTFbin);
                            [~, ~, tuningCurveTFconf(q,:), ~] = normfit(frChangeTFbin);

                            normFrTFcombTMP = [normFrTFcombTMP ; frChangeTFbin];
                            anovaLabel = [anovaLabel repmat(q,1,length(frChangeTFbin))];
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
                        resultsSes(cl).tuningCurveTF = NaN(1,size(TFbins,1));
                        resultsSes(cl).tuningCurveTFconf = NaN(size(TFbins,1),2);
                        resultsSes(cl).TFbinsMean = NaN(1,size(TFbins,1));    
                        resultsSes(cl).TFbinAvgFr = NaN(size(TFbins,1), (PSTHwindow(2)-PSTHwindow(1))/binSize);
                        resultsSes(cl).TFRespPeakTime = NaN;
                        resultsSes(cl).TFRespPeakHW = NaN;
                    end
%                 else
%                     resultsSes(cl).tuningCurveTF = NaN(1,size(TFbins,1));
%                     resultsSes(cl).tuningCurveTFconf = NaN(size(TFbins,1),2);
%                     resultsSes(cl).TFbinsMean = NaN(1,size(TFbins,1));    
%                     resultsSes(cl).TFbinAvgFr = NaN(size(TFbins,1), (PSTHwindow(2)-PSTHwindow(1))/binSize);
%                 end
            end
            
%             allUnitsSumm = updateFileds(allUnitsSumm, resultsSes, fields, ind);
            allUnitsSumm.TF = [allUnitsSumm.TF resultsSes];
            
        end
    end
    
end

time = toc;
disp(['Finished ' subjects{i} ' in ' num2str(time/60) ' min'])


end


function allUnitsSumm = updateFileds(allUnitsSumm, resultsSes, fields, ind)

for i=1:length(fields)
    for j = 1:length(ind)
        allUnitsSumm.TF(ind(j)).(fields{i}) = resultsSes(j).(fields{i});
    end
end
end

function [frTFnorm, sd] = calcZscore(fr, binSize, preEventWind, sd, varargin)

    frBase = mean(fr(:,1:round((preEventWind/binSize))),1);   %avg across trials
    if isempty(sd)
     sd = std(frBase);
    end
    frTFnorm = (fr - (mean(frBase)))/sd; 
    
    if isempty(varargin)
        frTFnorm(:, 1:round(preEventWind/binSize)) = [];  % leave only spikes after TF change onset
    end
end


function [startInd, endInd] = calcPeakHWind(normFr, peakInd)
    peakFr = normFr(peakInd);
    minHalfPeakWidth = 20;
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
    
    if endInd-startInd<minHalfPeakWidth 
        extra = floor((minHalfPeakWidth-(endInd-startInd))/2);
        startInd = startInd - extra;
        endInd = endInd + extra;
        if startInd<1
            startInd = 10;
%             endInd = minHalfPeakWidth;
        end
        if endInd>length(normFr)
            endInd=length(normFr);
%             startInd = length(normFr) - minHalfPeakWidth;
        end
           
    end
    
end

