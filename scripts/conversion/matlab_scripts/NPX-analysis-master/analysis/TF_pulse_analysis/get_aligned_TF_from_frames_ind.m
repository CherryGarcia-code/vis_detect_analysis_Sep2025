function TF_aligned = get_aligned_TF_from_frames_ind(TrialsData, frames_ind, window)

frames_numb_pre = ceil(-window(1)/0.0167);
frames_numb_post = ceil(window(2)/0.0167);
TF_aligned = [];

for tr = 1:length(frames_ind)
    if ~isempty(frames_ind{tr})
        frames_ind_tr = frames_ind{tr};
        TF_tr = TrialsData(tr).TF;
        TF_tr(TF_tr==0) = [];
        TF_aligned_tmp = [];
        
        for i = 1:length(frames_ind_tr)
            TF_aligned_tmp = [TF_aligned_tmp TF_tr(frames_ind_tr(i)-frames_numb_pre:frames_ind_tr(i)+frames_numb_post-1)];  
        end
        TF_aligned = [TF_aligned TF_aligned_tmp];
    end
end

TF_aligned = TF_aligned';
end

