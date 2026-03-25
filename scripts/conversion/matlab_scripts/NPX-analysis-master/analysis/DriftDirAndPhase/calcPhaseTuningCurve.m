function phaseTuningCurve = calcPhaseTuningCurve(spikeTimes, phaseBinStartTimesSes, phaseBinEndTimesSes, varargin)

% inputs are: 
% 1. spike times
% 2. cell arrays with format of DriftDir X number phase bins;
% within each cell are times of frame onsets that correspond to the phase 
% bin start/end times during the session

% the function than calculates average firing rate within each bin, averaged
% across all input trials

if isempty(varargin)
    averageTC = 1;  % default mode, average across each phase bin, get average phase tuning curve per drift direction
else 
    if strcmp(varargin{1}, 'events')==1 % don't average across each phase bin, usefull for ANOVA stats for example
        averageTC = 0;
    end
end

if averageTC==1    
    phaseTuningCurve = NaN(size(phaseBinStartTimesSes,1), size(phaseBinStartTimesSes,2));
    for g=1:size(phaseBinStartTimesSes,1) % drift direction
        for i=1:size(phaseBinStartTimesSes,2)
            phaseBinStartTimes = phaseBinStartTimesSes{g,i};
            phaseBinEndTimes = phaseBinEndTimesSes{g,i};

            spCount = 0;
            for j=1:length(phaseBinStartTimes)
                spCount = spCount + sum(spikeTimes>=phaseBinStartTimes(j)&spikeTimes<phaseBinEndTimes(j));
            end

            spCount = spCount/sum(phaseBinEndTimes-phaseBinStartTimes);    % go to Hz
            phaseTuningCurve(g,i) = spCount;
        end
    end
end

if averageTC==0    
    phaseTuningCurve = cell(size(phaseBinStartTimesSes,1), size(phaseBinStartTimesSes,2));
    for g=1:size(phaseBinStartTimesSes,1) % drift direction
        for i=1:size(phaseBinStartTimesSes,2)
            phaseBinStartTimes = phaseBinStartTimesSes{g,i};
            phaseBinEndTimes = phaseBinEndTimesSes{g,i};

            spCount = NaN(1,length(phaseBinStartTimes));
            for j=1:length(phaseBinStartTimes)
                spCount(j) = sum(spikeTimes>=phaseBinStartTimes(j)&spikeTimes<phaseBinEndTimes(j));
            end
            
            spCountDur = phaseBinEndTimes-phaseBinStartTimes;
            spCount = spCount./spCountDur;    % go to Hz
            phaseTuningCurve{g,i} = spCount;
        end
    end
end

 

end

