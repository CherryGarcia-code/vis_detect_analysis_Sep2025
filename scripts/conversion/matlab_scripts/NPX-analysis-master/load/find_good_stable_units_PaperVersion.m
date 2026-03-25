function good_and_stable_clusters = find_good_stable_units_PaperVersion(probe)

%Keep only stable units from Kilosort good units. Criteria are as in for naive mice data in Khilkevich&Lohse 2024
%   Detailed explanation goes here

good_clusters = probe.cluster_id_KS_good;
ses_tot_time = probe.st(end);

is_stable = zeros(1, length(good_clusters));
for cl = 1:length(good_clusters)
    sp_times = probe.st(probe.clu==good_clusters(cl));
    avg_fr = length(sp_times)/ses_tot_time;

    [~, ~, ~, ~, ~, spCounts] = psthAndBA(sp_times, 0, [0 ses_tot_time], 0.01); % from the start of recording, in 10ms bins
    fr20MinutesWind = movmean(spCounts/0.01, 20*60*100);
    fr10MinutesWind = movmean(spCounts/0.01, 10*60*100);
    fr5MinutesWind = movmean(spCounts/0.01, 5*60*100);

    ISIdistr = diff(sp_times);
    ISIdistrCounts1msBin = histcounts(ISIdistr,0:0.001:0.05);
    maxISIDistrIndFirst5ms = find(ISIdistrCounts1msBin(1:5)==max(ISIdistrCounts1msBin(1:5)));

    ISIdistrCounts1msBinSorted= sort(ISIdistrCounts1msBin, 'descend');
    firstISIDistrPeakHeight = ISIdistrCounts1msBinSorted(1);
    secondISIDistrPeakHeight = ISIdistrCounts1msBinSorted(2); % testing smoothness of distr, "second" peak should be close in size to the first one 

    if avg_fr>=0.5 & min(fr20MinutesWind)>=0.3*avg_fr & min(fr10MinutesWind)>=0.2*avg_fr & min(fr5MinutesWind)>=0.1*avg_fr & maxISIDistrIndFirst5ms>2 & firstISIDistrPeakHeight<4*secondISIDistrPeakHeight
         is_stable(cl) = 1;
    end
end

disp(['Found ' num2str(sum(is_stable)) ' stable clusters out of ' num2str(length(good_clusters)) ])
good_and_stable_clusters = good_clusters(is_stable==1);   

end

