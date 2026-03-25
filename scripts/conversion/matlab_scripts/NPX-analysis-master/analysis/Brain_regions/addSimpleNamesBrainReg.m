load('brain_reg_list_comb.mat');
subjects = fieldnames(data);

for i = 1:length(subjects)
    
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probes_numb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        
        for k = 1:probes_numb    
            good_and_stab_cl_coord = data.(subjects{i}).(sessions{j}).NPX_probes(k).good_and_stab_cl_coord;
            brain_region = {good_and_stab_cl_coord.brain_region};
            
            for q = 1:length(brain_region)
                brain_region_sel = brain_region{q};
                brain_reg_sel_new = findSimpleRegionName(brain_region_sel, brain_reg_list_comb);
                good_and_stab_cl_coord(q).brain_region_comb = brain_reg_sel_new; 
            end
            data.(subjects{i}).(sessions{j}).NPX_probes(k).good_and_stab_cl_coord = good_and_stab_cl_coord;
        end
    end
    
end