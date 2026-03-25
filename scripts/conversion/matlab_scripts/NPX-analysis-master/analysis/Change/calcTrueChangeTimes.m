
function newChangeTimes = calcTrueChangeTimes(TrialsData, hitTrials, missTrials, frameTimes, ChangeONtimes, BaselineONtimes)

isChangePresent = (hitTrials==1)|(missTrials==1);

for tr = 1:length(TrialsData)
    if (BaselineONtimes(tr)<frameTimes{tr}(1))||((frameTimes{tr}(1)-BaselineONtimes(tr))>0.1)   % condition fails on 0.1% of trials, unclear how/why 
        if isChangePresent(tr)
            try
                tag = TrialsData(tr).tag;
                baselineStartFrame = strfind(tag, 'B');
                baselineStartFrame = baselineStartFrame(1);
                changeStartFrame = strfind(tag, 'C');
                changeStartFrame = changeStartFrame(1);
                changeStartFrame = changeStartFrame-baselineStartFrame;
                newChangeTimes(tr) = frameTimes{tr}(changeStartFrame);
            catch
                newChangeTimes(tr) = NaN;
            end
        else
            newChangeTimes(tr) = NaN;
        end
    else
        newChangeTimes(tr) = ChangeONtimes(tr);
    end
end

end

    
