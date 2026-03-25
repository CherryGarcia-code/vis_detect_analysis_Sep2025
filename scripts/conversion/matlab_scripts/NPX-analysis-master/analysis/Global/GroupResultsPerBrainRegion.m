function [unitPerBrainReg, BrainRegGroups, BrainRegGroupNames, params] = GroupResultsPerBrainRegion(allUnitsSumm)

TFparams = allUnitsSumm.TFparams;
TFbins = TFparams.TFbins;
PSTHwindowTF = TFparams.PSTHwindow;
binSizeTF = TFparams.binSize;
delayBTWpulses = allUnitsSumm.TFseqparams.delayBTWpulses;
delayBTWpulsesPlot = [-0.05 delayBTWpulses-0.05];

phaseBinsPlot = allUnitsSumm.PhaseParams.phaseBinsPlot;
phaseBins = allUnitsSumm.PhaseParams.phaseBins;

driftDir = allUnitsSumm.PhaseParams.driftDir;

brainRegions = {allUnitsSumm.Units.brainRegionComb};
avgFR = [allUnitsSumm.Units.avgFR];

pValuesTF = [allUnitsSumm.TF.tuningCurveTFpval];
TFRespPeakTime = [allUnitsSumm.TF.TFRespPeakTime];
TFRespPeakHW = [allUnitsSumm.TF.TFRespPeakHW];

tuningCurvesTF = reshape([allUnitsSumm.TF.tuningCurveTF], [], length(pValuesTF))';
TFbinsMean = nanmean(reshape([allUnitsSumm.TF.TFbinsMean], [], length(pValuesTF))');
TFbinsPlot = TFbinsMean(2:end-1);

TFbinAvgFr = reshape([allUnitsSumm.TF.TFbinAvgFr], length(TFbinsMean), [], length(pValuesTF));
TFAvgResp = TFbinAvgFr(end-1,:,:)-TFbinAvgFr(2,:,:);
TFAvgResp = permute(TFAvgResp, [3 2 1]);

TFbinAvgFr = TFbinAvgFr(2:end-1, :, :);

TFSeqIncrPeakVal = reshape([allUnitsSumm.TF.TFSeqIncrPeakVal], [], length(pValuesTF))';

PSTHwindowEL = allUnitsSumm.EarlyLickParams.PSTHwindow;
ZscoreThreshEL = allUnitsSumm.EarlyLickParams.ZscoreThresh;
ZscoreELAvg = reshape([allUnitsSumm.EarlyLick.ZscoreELAvg], [], length(pValuesTF))';
hasELRamp = [allUnitsSumm.EarlyLick.hasELRamp];

pvalDriftDir = reshape([allUnitsSumm.Phase.pvalDriftDir], [], length(pValuesTF))';
phaseTuningCurves = reshape([allUnitsSumm.Phase.phaseTuningCurves], length(phaseBinsPlot), [], length(pValuesTF));
phaseTuningCurves = permute(phaseTuningCurves, [3 1 2]);

brainRegionsRec = unique(brainRegions);
unitPerBrainReg = [];
minFRthresh = 0.5;

for i = 1:length(brainRegionsRec)
    ind = find(strcmp(brainRegions, brainRegionsRec{i}));
    ind(avgFR(ind)<minFRthresh) = []; 

    unitPerBrainReg(i).name = brainRegionsRec{i}; 
    unitPerBrainReg(i).TotRecUnits = length(ind); 
    unitPerBrainReg(i).pValuesTF = pValuesTF(ind);
    unitPerBrainReg(i).tuningCurvesTF = tuningCurvesTF(ind,:);
    unitPerBrainReg(i).TFRespPeakTime = TFRespPeakTime(ind);
    unitPerBrainReg(i).TFRespPeakHW = TFRespPeakHW(ind);
    unitPerBrainReg(i).TFAvgResp = TFAvgResp(ind, :);
    unitPerBrainReg(i).TFSeqIncrPeakVal = TFSeqIncrPeakVal(ind,:);
    unitPerBrainReg(i).TFbinAvgFr = TFbinAvgFr(:,:,ind);
    unitPerBrainReg(i).ZscoreELAvg = ZscoreELAvg(ind, :);
    unitPerBrainReg(i).hasELRamp = hasELRamp(ind);
    unitPerBrainReg(i).pvalDriftDir = pvalDriftDir(ind, :); 
    unitPerBrainReg(i).phaseTuningCurves = phaseTuningCurves(ind,:,:);
    unitPerBrainReg(i).avgFR = avgFR(ind);
    
end

tooFewUnits = find([unitPerBrainReg.TotRecUnits]<10);
unitPerBrainReg(tooFewUnits) = [];

[BrainRegGroups, BrainRegGroupNames] = defineBrainRegGroups;

params.PSTHwindowTF = PSTHwindowTF;
params.binSizeTF = binSizeTF;
params.delayBTWpulsesPlot = delayBTWpulsesPlot;
params.TFbinsPlot = TFbinsPlot;
params.PSTHwindowEL = PSTHwindowEL;
params.ZscoreThreshEL = ZscoreThreshEL;
params.driftDir = driftDir;
params.phaseBinsPlot = phaseBinsPlot;
params.phaseBins = phaseBins;

end

