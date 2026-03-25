function brain_reg_new = findSimpleRegionName(brain_reg, brain_reg_list_comb)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

ind = [];
for i=1:length(brain_reg_list_comb)
    if startsWith(brain_reg, brain_reg_list_comb{i})==1
        ind = [ind i];
    end
end
% if length(ind)>1||isempty(ind)  
%     if sum(startsWith(brain_reg,{'PL', 'ACA','cc','cing','cpd','PP','MG','PIL', 'CA','AP','PRN','VISp','LH','PA', 'AI', 'VAL','V','VPM','VM','MEA','fiber','P','RSP','PoT','SG','III','SS','ar','SI','AN','CU','LA','st'})==0)
%            a=1; 
%     end
% end
    brain_reg_list_tmp = brain_reg_list_comb(ind);
    brain_new_tmp = cellfun(@length, brain_reg_list_tmp);
    if sum(strcmp(brain_reg_list_tmp, brain_reg))>0
        best_guess_ind = find(strcmp(brain_reg_list_tmp, brain_reg)==1);
    else
        best_guess_ind = find(brain_new_tmp==max(brain_new_tmp));
    end
    brain_reg_new = brain_reg_list_tmp{best_guess_ind};
end

