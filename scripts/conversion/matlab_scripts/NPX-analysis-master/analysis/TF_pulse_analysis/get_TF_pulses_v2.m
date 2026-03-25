function [TF_frame_times, TF_frame_ind] = get_TF_pulses_v2(TrialsData, Change_ON_dur, frame_times, MotionOnsetTimes, pulse_thresh_min, pulse_thresh_max, thresh_scale, post_event_wind, trial_ind_to_use, varargin)

%  Andrei Khilkevich 2021
%  Get absolute times and indexes of frames during which TF pulses within
%  pulse_thresh_min<x<pulse_thresh_max range have occured (for linear scale)

%  By default: exclude 1s from baseline onset, 
%  2s before lick registration(or lick onset if motion onset estimation was done - cleans up responses in some units),
%  and exclude all change periods. 


if isempty(varargin)
    baseline_period_used = NaN; % use full baseline
    del_time_before_FA_or_Ab = 2;
    del_time_from_Base_onset = 1;
    IFI = 0.0167;
else
    if length(varargin)==1
        varargin = varargin{1};
    end
    baseline_period_used = varargin{1};
    del_time_before_FA_or_Ab = varargin{2};
    del_time_from_Base_onset = varargin{3};
    IFI = varargin{4};
end

del_frames_from_Base_onset = ceil(del_time_from_Base_onset/IFI);

if ~isnan(baseline_period_used)
    if del_time_from_Base_onset<baseline_period_used(1)
        del_frames_from_Base_onset = ceil(baseline_period_used(1)/IFI);
    end
    max_frame_numb = ceil(baseline_period_used(2)/IFI);
else
    max_frame_numb = Inf;
end    
    
for i = 1:length(trial_ind_to_use)
    tr = trial_ind_to_use(i);
    TF_tr = TrialsData(tr).TF;
    TF_tr(TF_tr==0) = [];
       
    if  (TrialsData(tr).IsHit==1) || (TrialsData(tr).IsMiss==1)
        frames_to_del = ceil( (Change_ON_dur(tr)+post_event_wind)/IFI);
        try
            TF_tr(end-frames_to_del+1:end) = [];        % cut off TF during changeON  
            if length(TF_tr)>max_frame_numb
                TF_tr(max_frame_numb+1:end) = [];  
            end
        catch
            TF_tr = [];
        end
    else
        if ( TrialsData(tr).IsFA==1 )||( TrialsData(tr).IsAbort==1 )    
            if ~isempty(MotionOnsetTimes)
                MotionOnsetTime = MotionOnsetTimes(tr);
                framesToDelInd = find(frame_times{tr}>=(MotionOnsetTime-(del_time_before_FA_or_Ab+post_event_wind)), 1, 'first'); % % don't use frames in (def 2s) before FA or abort
            else
                frames_to_del = ceil(del_time_before_FA_or_Ab/IFI);   % don't use frames in (def 2s) before FA or abort
            end
        try
            if ~isempty(MotionOnsetTimes)
                TF_tr(framesToDelInd:end) = [];
            else
                TF_tr(end-frames_to_del+1:end) = [];
            end
            if length(TF_tr)>max_frame_numb
                TF_tr(max_frame_numb+1:end) = [];  
            end
        catch           % if there were too few frames
            TF_tr = [];
        end
        else
            TF_tr = [];
        end
    end
    
    TF_tr_parsed{i} = TF_tr;
end

TF_all = cell2mat(TF_tr_parsed(:));
TF_std = std(TF_all);

if strcmp(thresh_scale, 'log2')
    pulse_threshTF_min = 2^(pulse_thresh_min*TF_std);
    pulse_threshTF_max = 2^(pulse_thresh_max*TF_std);
else
    if strcmp(thresh_scale, 'lin')
        pulse_threshTF_min = pulse_thresh_min;
        pulse_threshTF_max = pulse_thresh_max;    
    end
end

for i = 1:length(TF_tr_parsed)
    tr = trial_ind_to_use(i);
    
    if  length(TF_tr_parsed{i})>del_frames_from_Base_onset
        
        TF_ind_tr = find((TF_tr_parsed{i}>=pulse_threshTF_min)&(TF_tr_parsed{i}<pulse_threshTF_max));
        TF_ind_tr = leave_pulse_start_ind(TF_ind_tr, TF_tr_parsed{i});
        TF_ind_tr(find(TF_ind_tr<=del_frames_from_Base_onset)) = [];
    else
        TF_ind_tr = [];
    end
    
    TF_frame_ind{i} = TF_ind_tr;
    try
        TF_frame_times{i} = frame_times{tr}(TF_ind_tr);
    end
end

    function pulse_start_ind = leave_pulse_start_ind(pulse_ind, TF)
        if length(pulse_ind)>=3
            pulse_start_ind =  [pulse_ind(1) ; pulse_ind(find(diff(TF(pulse_ind))~=0)+1)];
        else
            pulse_start_ind = [];
        end
    end

end

