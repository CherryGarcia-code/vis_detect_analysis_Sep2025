function  frTrType = calcFR(spikeTimes, EventTimes, tBin, PSTHwindow, sigma, trimFr, ChangeONtimes, varargin)
                
if isempty(varargin)
    mode = 'avg';
else
    if strcmp(varargin{1}, 'trials')
        mode = 'trials';
    end
end

minTrialNumb = 10;

if length(EventTimes)>=minTrialNumb
    
    psthExtra = 0.5; 
    [~, ~, ~, ~, ~, frTr] = psthAndBA(spikeTimes, EventTimes, [PSTHwindow(1)-psthExtra PSTHwindow(2)+psthExtra], tBin);

    gaussHw = 4*sigma;
    x = -gaussHw:tBin:gaussHw;
    gaussWindow = normpdf(x, 0, sigma);
    gaussWindow  = gaussWindow./sum(gaussWindow);

    if trimFr == 1     % replace fr after Change onset with NaNs
        for tr = 1:length(ChangeONtimes)   % intended to be used for BaselineON events alignment
            if ~isnan(ChangeONtimes(tr))
                Change_ON_ind =   int32(1+(-PSTHwindow(1) + psthExtra +(ChangeONtimes(tr) - EventTimes(tr)))/tBin);
                if Change_ON_ind<length(frTr(1,:))
                    frTr(tr, Change_ON_ind:end) = NaN;
                end
            end
        end
    end    
    
    if strcmp(mode, 'avg')
        frTrType = nanmean(frTr, 1)/tBin;
        if sigma>0
            frTrType = conv(frTrType, gaussWindow, 'same');
        end
    else
        if strcmp(mode, 'trials')
            frTrType = frTr/tBin;
        if sigma>0
            frTrType = conv2(frTrType, gaussWindow, 'same');
        end
        end
    end
        
    frTrType(:, [1:psthExtra/tBin end-psthExtra/tBin+1:end]) = [];
    
else
    frTrType = NaN(1, (PSTHwindow(2) - PSTHwindow(1))/tBin );
end
    

end

