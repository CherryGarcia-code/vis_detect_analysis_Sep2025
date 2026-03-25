function [TF_frame_times, TF_frame_ind] = get_TF_pulses_seq(TrialsData, Change_ON_dur, frame_times, MotionOnsetTimes, pulse_thresh_seq, delay_btw_pulses, thresh_scale, post_event_wind, trial_ind_to_use, varargin)

%  Andrei Khilkevich 2021
%  Get absolute times and indexes of frames during which a sequence of TF pulses A->B have occured.
%  The range for A and B is defined in pulse_thresh_seq, as
%  pulse_thresh_seq{1}(1) < A < pulse_thresh_seq{1}(2)
%  pulse_thresh_seq{2}(1) < B < pulse_thresh_seq{2}(2)  (for linear scale)

%  By default, exclude cases of events during: 1s from baseline onset, 
%  2s before lick registration(or lick onset if motion onset estimation was done - cleans up responses in some units),
%  and exclude all change periods. 


if isempty(varargin)
    baseline_period_used = NaN; % use full baseline
    del_time_before_FA_or_Ab = 2;
    del_time_from_Base_onset = 1;
    IFI = 0.0167;
else
    varargin = varargin{1};
    baseline_period_used = varargin{1};
    del_time_before_FA_or_Ab = varargin{2};
    del_time_from_Base_onset = varargin{3};
    IFI = varargin{4};
end

max_Nframes_btw_pulses = ceil(delay_btw_pulses/IFI);
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

for seq = 1:length(pulse_thresh_seq)
    if strcmp(thresh_scale, 'log2')
        pulse_threshTF_min(seq) = 2^(pulse_thresh_seq{seq}(1)*TF_std);
        pulse_threshTF_max(seq) = 2^(pulse_thresh_seq{seq}(2)*TF_std);
    else
        if strcmp(thresh_scale, 'lin')
            pulse_threshTF_min(seq) = pulse_thresh_seq{seq}(1);
            pulse_threshTF_max(seq) = pulse_thresh_seq{seq}(2);
        end
    end
end

for i = 1:length(TF_tr_parsed)
    tr = trial_ind_to_use(i);
    
    if  length(TF_tr_parsed{i})>del_frames_from_Base_onset&&length(TF_tr_parsed{i})>max_Nframes_btw_pulses
        
        TF_1st_cond = (TF_tr_parsed{i}>=pulse_threshTF_min(1))&(TF_tr_parsed{i}<pulse_threshTF_max(1));
        TF_1st_cond(end-max_Nframes_btw_pulses+1:end) = []; 
        TF_2nd_cond = (TF_tr_parsed{i}>=pulse_threshTF_min(2))&(TF_tr_parsed{i}<pulse_threshTF_max(2));        
        TF_2nd_cond(1:max_Nframes_btw_pulses) = [];
        TF_ind_tr = find(TF_1st_cond==1&TF_2nd_cond==1);

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
%             pulse_start_ind =  [pulse_ind(1) ; pulse_ind(find(diff(pulse_ind)>1)+1)];
            pulse_start_ind =  [pulse_ind(1) ; pulse_ind(find(diff(TF(pulse_ind))~=0)+1)];
        else
            pulse_start_ind = [];
        end
    end

end

