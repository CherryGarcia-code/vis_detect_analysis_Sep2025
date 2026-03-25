function allUnitsSumm = createMainSumStruct(data)

probeNumbTot = 0;
probeCounter = 0;
subjects = fieldnames(data);

for i = 1:length(subjects)
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probesNumb = length(data.(subjects{i}).(sessions{j}).NPX_probes);
        probeNumbTot = probeNumbTot + probesNumb;
    end
end

allUnitsSumm.AnimalsOrder = subjects;
allUnitsSumm.Units = [];

for i = 1:length(subjects)
    
    sessions = fieldnames(data.(subjects{i}));
    for j = 1:length(sessions)
        probesNumb = length(data.(subjects{i}).(sessions{j}).NPX_probes);

        for p = 1:probesNumb
            probeCounter = probeCounter + 1;
            waitbar(probeCounter/probeNumbTot)
            
            good_clusters = data.(subjects{i}).(sessions{j}).NPX_probes(p).cluster_id_good_and_stable;
            good_cl_coord = data.(subjects{i}).(sessions{j}).NPX_probes(p).good_and_stab_cl_coord;
            resultsSes = [];

            for cl = 1:length(good_clusters)
                resultsSes(cl).subjInd = i;
                resultsSes(cl).sesInd = j;
                resultsSes(cl).probeInd = p;
                resultsSes(cl).cluInd = good_clusters(cl);
                resultsSes(cl).brainRegion = good_cl_coord(cl).brain_region;
                resultsSes(cl).brainRegionComb = good_cl_coord(cl).brain_region_comb;
                resultsSes(cl).AP = good_cl_coord(cl).y;
                resultsSes(cl).ML = good_cl_coord(cl).x;
                resultsSes(cl).DV = good_cl_coord(cl).z;
            end
            allUnitsSumm.Units = [allUnitsSumm.Units resultsSes];
        end
    end
end

end



