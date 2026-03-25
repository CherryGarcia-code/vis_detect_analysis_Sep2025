function  [phaseBinStartTimes, phaseBinEndTimes] = getBinnedPhaseTimes(TrialsData, Change_ON_dur, frame_times, phaseBins, trial_ind_to_use, driftDirGroup)

del_time_before_FA_or_Ab = 2;
del_time_from_Base_onset = 1;
IFI = 0.0167;
        
phase = {TrialsData.phase};
StimOri = [TrialsData.Stim1Ori];
del_frames_from_Base_onset = ceil(del_time_from_Base_onset/IFI);

for i = 1:length(trial_ind_to_use)
    
    tr = trial_ind_to_use(i);
    phaseTr = phase{tr}(:,1);
    phaseTr(phaseTr==0) = [];
    phaseTr = [0 ; phaseTr]; % 1st frame has zero deg phase
    phaseTr  = mod(phaseTr, 360);
    
    if StimOri(tr) == driftDirGroup(1)       % group trials by drift direction
        g=1;
    else
        if StimOri(tr) == driftDirGroup(2)
            g=2;
        end
    end
    
    if  (TrialsData(tr).IsHit==1) || (TrialsData(tr).IsMiss==1)
        frames_to_del = ceil( (Change_ON_dur(tr)+0.5)/IFI);
        try
            phaseTr(end-frames_to_del+1:end) = [];        % cut off TF during changeON  
        catch
            phaseTr = [];
        end
    else
        if ( TrialsData(tr).IsFA==1 )||( TrialsData(tr).IsAbort==1 )    
            frames_to_del = ceil(del_time_before_FA_or_Ab/IFI);   % don't use frames in (default 2s) before FA or abort
            try
                phaseTr(end-frames_to_del+1:end) = [];    
            catch           % if there were too few frames
                phaseTr = [];
            end
        else
            phaseTr = [];
        end
    end    
    
    try
        phaseTr(1:del_frames_from_Base_onset) = [];
        frameTimesTr = frame_times{tr};
        frameTimesTr(1:del_frames_from_Base_onset)=[];
    catch
        phaseTr = [];
    end
    
    if ~isempty(phaseTr)
        for j=1:length(phaseBins)-1       
            phaseBinInd = find(phaseTr>=phaseBins(j)&phaseTr<phaseBins(j+1));
            if ~isempty(phaseBinInd)
                [BinStartInd, BinEndInd] = getSequentialIndStartEnd(phaseBinInd);
                try
                   phaseBinStartTimes{g,j} = [phaseBinStartTimes{g,j} frameTimesTr(BinStartInd)];
                   phaseBinEndTimes{g,j} = [phaseBinEndTimes{g,j} frameTimesTr(BinEndInd)];
                catch
                   phaseBinStartTimes{g,j} = frameTimesTr(BinStartInd);
                   phaseBinEndTimes{g,j} = frameTimesTr(BinEndInd);
                end
            end
        end
    end
    
end



end


function [BinStartInd, BinEndInd] = getSequentialIndStartEnd(phaseBinInd)
    ind = find(diff(phaseBinInd)>1);
    if ~isempty(ind)
        BinStartInd =  [phaseBinInd(1); phaseBinInd(ind+1)];
        BinEndInd =  [phaseBinInd(ind)+1; phaseBinInd(end)+1];
    else
        BinStartInd = phaseBinInd(1);
        BinEndInd = phaseBinInd(end)+1;
    end
end

