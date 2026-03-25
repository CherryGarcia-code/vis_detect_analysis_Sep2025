
function plotTFTuning(allUnitsSumm)

BrainRegGroups = defineBrainRegGroups;

[unitPerBrainReg, BrainRegGroups, params] = GroupResultsPerBrainRegion(allUnitsSumm);

PSTHwindow = params.PSTHwindow;
binSize = params.binSize;
TFbinsPlot = params.TFbinsPlot;
delayBTWpulsesPlot = params.delayBTWpulsesPlot;

% 
% %%
% pValThresh = 0.01;
% for i = 1:length(unitPerBrainReg)
%     unitPerBrainReg(i).fracSignUnits = sum(unitPerBrainReg(i).pValues<pValThresh)/length(unitPerBrainReg(i).pValues);
% end
% 
% [fracSignUnitsSorted, fracSignUnitsSortedind] = sort([unitPerBrainReg.fracSignUnits], 'descend');
% brainRegSorted =  {unitPerBrainReg(fracSignUnitsSortedind).name};
% totRecUnitsSorted = [unitPerBrainReg(fracSignUnitsSortedind).TotRecUnits];
% 
% figure('units','normalized','outerposition',[0 0.1 1 0.85]);
% columnsNumb = 25;
% rowsNumb = ceil(length(fracSignUnitsSortedind)/columnsNumb);
%  
% for row = 1:rowsNumb
%     subplot(rowsNumb, 1, row) 
%     hold on
%     
%     if row*columnsNumb<=length(fracSignUnitsSorted)
%         indPlot = 1+columnsNumb*(row-1):row*columnsNumb;
%     else
%         indPlot = 1+columnsNumb*(row-1):length(fracSignUnitsSorted);
%     end
%     
%     brainRegSorteRow = brainRegSorted(indPlot);
%     bar(1:length(indPlot), fracSignUnitsSorted(indPlot), 'k'); 
%     
%     if sum(fracSignUnitsSorted(indPlot)<=pValThresh)>0
%         indBelowChance = find(fracSignUnitsSorted(indPlot)<=pValThresh);
%         bar(indBelowChance, fracSignUnitsSorted(indPlot(indBelowChance)), 'FaceColor', [0.4 0.4 0.4]); 
%     end
%         
%     text(1:length(indPlot), fracSignUnitsSorted(indPlot), num2str(totRecUnitsSorted(indPlot)'),'vert','bottom','horiz','center'); 
% 
%     if row == 1
%        ylim_all = ylim; % use the same y scale for al rows
%     end
%     
%     xticks(1:length(indPlot))
%     xticklabels(brainRegSorteRow);
%     ylim(ylim_all);
%     box off
%     if row ==2
%         ylabel(['Fraction with significant TF tuning (p<' num2str(pValThresh, 2) ')'], 'FontSize', 14)
%     end
% end


%% distributions of peak time and width of TF responses
pValThresh = 0.01;

figure('units','normalized','outerposition',[0.1 0.1 0.6 0.6]);
TFBins = 0:20:700;
distrAll = [];

brNamesAll = [];
meanDistrAll = [];
meanDistrAllHW = [];
confAll = [];

for i = 1:length(BrainRegGroups)
    brRegGroup = BrainRegGroups{i};
%     brRegGroup = brainRegionsRec;
    brRegGroup = leaveBrRegWithEnoughUnits({unitPerBrainReg.name}, brRegGroup, {unitPerBrainReg.pValues}, pValThresh, 10);
    for j = 1:length(brRegGroup)
        brRegOfIntr = brRegGroup{j};
        brRegInd = find(strcmp({unitPerBrainReg.name}, brRegOfIntr)==1);
        signUnitsInd = find(unitPerBrainReg(brRegInd).pValues<pValThresh);
        TFRespPeakTime = unitPerBrainReg(brRegInd).TFRespPeakTime(signUnitsInd);
        TFRespHalfPeakW = unitPerBrainReg(brRegInd).TFRespPeakHW(signUnitsInd);

%         [distrBrReg binsPlot] = histcounts(TFRespPeakTime, TFBins, 'Normalization', 'probability');
        [distrBrReg binsPlot] = histcounts(TFRespHalfPeakW, TFBins, 'Normalization', 'probability');

        distrAll = [distrAll; distrBrReg];
        meanDistrAll = [meanDistrAll median(TFRespHalfPeakW)];
        conf = bootci(5000, @median, TFRespHalfPeakW);
%         meanDistrAll = [meanDistrAll median(TFRespPeakTime)];
%         conf = bootci(5000, @median, TFRespPeakTime);

        meanDistrAllHW = [meanDistrAllHW median(TFRespHalfPeakW)];
        
        confAll = [confAll conf];
    end
    brNamesAll = [brNamesAll brRegGroup];
end

PeakInd = [];
for i=1:length(brNamesAll)
    PeakInd(i) = find(distrAll(i,:)==max(distrAll(i,:)), 1, 'first');
end

[~, plotOrder] = sort(meanDistrAll);
% plotOrder=1:length(meanDistrAll);
imagesc(binsPlot, 1:size(distrAll,1), distrAll(plotOrder, :))
set(gca, 'YTick', 1:size(distrAll,1), 'YTickLabel', brNamesAll(plotOrder)) 
xlabel('TF response peak time, ms', 'FontSize', 14, 'FontName', 'Arial')
xlabel('TF response half-peak width, ms', 'FontSize', 14, 'FontName', 'Arial')
xlim([0 310])
colormap('hot')
h = colorbar;
ylabel(h, 'Fraction', 'FontSize', 14, 'FontName', 'Arial')

box off;

% 
% figure('units','normalized','outerposition',[0.1 0.1 0.3 0.5]);
% scatter(meanDistrAll, meanDistrAllHW, 50, 'k', 'filled')
% 
% xlabel('Median TF response peak time, ms', 'FontSize', 14, 'FontName', 'Arial')
% ylabel('Median TF response half-peak width, ms', 'FontSize', 14, 'FontName', 'Arial')
% [~, p] = corrcoef(meanDistrAll, meanDistrAllHW);
% title(['p = ' num2str(p(1,2))], 'FontWeight', 'normal')
% box off;







figure('units','normalized','outerposition',[0.1 0.1 0.6 0.6]);
hold on
set(gca, 'LineWidth', 1);
plot(meanDistrAll(plotOrder(end:-1:1)), 1:length(meanDistrAll), 'k' , 'LineWidth', 2)
scatter(meanDistrAll(plotOrder(end:-1:1)), 1:length(meanDistrAll), 100, 'k', 'filled' )
set(gca, 'YTick', 1:length(meanDistrAll), 'YTickLabel', brNamesAll(plotOrder(end:-1:1))) 

plot(confAll(1,plotOrder(end:-1:1)), 1:length(meanDistrAll), 'b', 'LineWidth', 0.5)
plot(confAll(2,plotOrder(end:-1:1)), 1:length(meanDistrAll), 'b', 'LineWidth', 0.5)
ylim([0.5 length(meanDistrAll)+0.5])
xlabel('Median TF response peak time, ms', 'FontSize', 14, 'FontName', 'Arial')
% xlabel('Median TF response half-peak width, ms', 'FontSize', 14, 'FontName', 'Arial')





%% CFDs of distributions of peak time and width of TF responses
pValThresh = 0.01;

figure('units','normalized','outerposition',[0.1 0.1 0.6 0.6]);
colors = lines(45);
count = 0;
for i = 1:length(BrainRegGroups)
    brRegGroup = BrainRegGroups{i};
    brRegGroup = leaveBrRegWithEnoughUnits({unitPerBrainReg.name}, brRegGroup, {unitPerBrainReg.pValues}, pValThresh, 10);
    subplot(3,4,i)
    hold on
    for j = 1:length(brRegGroup)
        brRegOfIntr = brRegGroup{j};
        count = count+1;
        brRegInd = find(strcmp({unitPerBrainReg.name}, brRegOfIntr)==1);
        signUnitsInd = find(unitPerBrainReg(brRegInd).pValues<pValThresh);
        TFRespPeakTime = unitPerBrainReg(brRegInd).TFRespPeakTime(signUnitsInd);
        TFRespPeakHW = unitPerBrainReg(brRegInd).TFRespPeakHW(signUnitsInd);
%         TFRespPeakHW(TFRespPeakTime<50) = [];
%         TFRespPeakTime(TFRespPeakTime<50) = [];
        
        [cdfPlot, PeakTimesPlot] = ecdf(TFRespPeakHW);
%         [cdfPlot, PeakTimesPlot] = ecdf(TFRespPeakTime);
        plot(PeakTimesPlot, cdfPlot, 'color', colors(count, :))

%         xlabel('Response Peak time, ms')
        ylabel('Fraction')
        xlabel('Peak half-width')
    end
    xlim([0 500])
    legend(brRegGroup, 'FontSize', 12, 'Location', 'Best');
    legend box off
end
%%
pValThresh = 0.01;

figure('units','normalized','outerposition',[0.1 0.1 0.6 0.6]);
colors = lines(45);
count = 0;
rsqBins = 0:0.01:1;


for i = 1:length(BrainRegGroups)
    brRegGroup = BrainRegGroups{i};
    brRegGroup = leaveBrRegWithEnoughUnits({unitPerBrainReg.name}, brRegGroup, {unitPerBrainReg.pValues}, pValThresh, 10);
    subplot(3,4,i)
    hold on
    for j = 1:length(brRegGroup)
        brRegOfIntr = brRegGroup{j};
        count = count+1;
        
        rAll{count} = [];
        rKendAll{count} = [];


        brRegInd = find(strcmp({unitPerBrainReg.name}, brRegOfIntr)==1);
        signUnitsInd = find(unitPerBrainReg(brRegInd).pValues<pValThresh);
        tuningCurvesTF = reshape([unitPerBrainReg(brRegInd).tuningCurvesTF(signUnitsInd, 2:end-1)], [], length(TFbinsPlot));
        rsq = [];
        rKend = [];
        r = [];
        
        for k=1:size(tuningCurvesTF,1)
            rKend(k) = abs(corr(tuningCurvesTF(k,:)', TFbinsPlot', 'Type','Kendall'));
            r(k) = abs(corr(tuningCurvesTF(k,:)', TFbinsPlot'));
%             rsq(k) = r.^2;
        end
        
        rAll{count} = r;
        rKendAll{count} = rKend;

        [cdfPlot, rsqCDF] = ecdf(r);
        plot(rsqCDF, cdfPlot, 'color', colors(count, :))
%         scatter(r, rKend, 10,  colors(count, :), 'filled')
%         ylabel('|r| Kendall')
%         xlabel('|r| Pearson]')
        
%         histogram(rsq, rsqBins, 'DisplayStyle', 'stairs', 'Normalization', 'Probability', 'EdgeColor', colors(count, :), 'LineWidth', 1)
        if (j - floor(j/3))==1
            ylabel('Fraction')
        end
        if j>8
            xlabel('|r| Person ')
        end
    end
    xticks([0:0.25:1]);
    xlim([0 1])
% axis([-1 1 -1 1])
    legend(brRegGroup, 'FontSize', 13, 'Location', 'Best');
    legend box off
end
%         scatter(cellfun(@mean, rKendAll), cellfun(@mean, rAll), 10, 'filled', 'k')
%%

pValThresh = 0.05;
zeroBinInd = 6;
fastTFBinInd = 11;
    
for i = 6%length(BrainRegGroups)%1:length(BrainRegGroups)
brRegGroup = BrainRegGroups{i};
brRegGroup = leaveBrRegWithEnoughUnits({unitPerBrainReg.name}, brRegGroup, {unitPerBrainReg.pValues}, pValThresh, 2);

figure('units','normalized','outerposition',[0 0 1 1]);

for j = 1:length(brRegGroup)
    subplot(2, length(brRegGroup), j)
    brRegOfIntr = brRegGroup{j};

    brRegInd = find(strcmp({unitPerBrainReg.name}, brRegOfIntr)==1);
    signUnitsInd = find(unitPerBrainReg(brRegInd).pValues<pValThresh);
    totUnits = unitPerBrainReg(brRegInd).TotRecUnits;

    tuningCurvesTF = reshape([unitPerBrainReg(brRegInd).tuningCurvesTF(signUnitsInd, 2:end-1)], [], length(TFbinsPlot));
%     tuningCurvesTF = reshape([unitPerBrainReg(brRegInd).tuningCurvesTF(signUnitsInd, :)], [], length(TFbinsPlot));
    
    [~, ind] = sort(tuningCurvesTF(:,fastTFBinInd), 'descend');
    tuningCurvesTF = tuningCurvesTF(ind,: );

    % figure('units','normalized','outerposition',[0.2 0.2 0.3 0.4]);
    imagesc(TFbinsPlot, 1:length(signUnitsInd), tuningCurvesTF)

    set(gca,'ytick',[])
    maxDev = sort([abs(min(tuningCurvesTF')) abs(max(tuningCurvesTF'))]);
%     maxDev = max([abs(min(tuningCurvesTF(:))) abs(max(tuningCurvesTF(:)))]);
    caxis([-maxDev(end-2) maxDev(end-2)])
    % caxis([-1.8 1.8]);

    % colormap('jet')
    h = colorbar;
    ylabel(h, 'Change in Z-score', 'FontSize', 16, 'FontName', 'Arial')
    box off;
    xlabel('Mean TF, Hz', 'FontSize', 16, 'FontName', 'Arial')
    title([brRegOfIntr ', ' num2str(round(100*length(signUnitsInd)/totUnits)) '% out of ' num2str(totUnits)], 'FontSize', 16, 'FontName', 'Arial')

    % simple average, units with larger effect on tuning contribute more
    
    tuningCurvesTFPlot = tuningCurvesTF;
%     tuningCurvesTFPlot = tuningCurvesTFPlot./max(abs(tuningCurvesTFPlot(:,end-1:end))')';
%     tuningCurvesTFPlot = tuningCurvesTFPlot./abs(tuningCurvesTFPlot(:,fastTFBinInd));

    indIncrToFast = find(mean(tuningCurvesTFPlot(:,fastTFBinInd:fastTFBinInd),2)>0);%tuningCurvesTFPlot(:,zeroBinInd));
    indDecrToFast = find(mean(tuningCurvesTFPlot(:,fastTFBinInd:fastTFBinInd),2)<0);%tuningCurvesTFPlot(:,zeroBinInd));

    [~, ~, tuningIncrToFastConf, ~] = normfit(tuningCurvesTFPlot(indIncrToFast,:));
    [~, ~, tuningDecrToFastConf, ~] = normfit(tuningCurvesTFPlot(indDecrToFast,:));

    % figure('units','normalized','outerposition',[0.2 0.2 0.2 0.4]);
    
    subplot(2, length(brRegGroup), length(brRegGroup)+j)
    hold on
    plot([TFbinsPlot(1) TFbinsPlot(end)], [0 0], 'k--')
    plot([1.02 1.02], [-15 15], 'k--')

    if length(indIncrToFast)>=5
        plot(TFbinsPlot, mean(tuningCurvesTFPlot(indIncrToFast, :)), 'k', 'LineWidth', 2)
%         scatter(TFbinsPlot, mean(tuningCurvesTFPlot(indIncrToFast, :)), 50, 'filled', 'k')
%         scatter(TFbinsPlot([1 end]), mean(tuningCurvesTFPlot(indIncrToFast, [1 end])), 100, 'filled', 'k')
%         plot(TFbinsPlot, tuningCurvesTFPlot(indIncrToFast, :), 'k', 'LineWidth', 0.5)
        ciplot(tuningIncrToFastConf(1,:), tuningIncrToFastConf(2,:), TFbinsPlot, 'k', 0.3)
    else
        tuningIncrToFastConf = zeros(size(tuningIncrToFastConf,1), size(tuningIncrToFastConf,2));
    end

    if length(indDecrToFast)>=5
        plot(TFbinsPlot, mean(tuningCurvesTFPlot(indDecrToFast, :)), 'r', 'LineWidth', 2)
%         scatter(TFbinsPlot, mean(tuningCurvesTFPlot(indDecrToFast, :)), 50, 'filled', 'r')
%         scatter(TFbinsPlot([1 end]), mean(tuningCurvesTFPlot(indDecrToFast, [1 end])), 100, 'filled', 'r')
%         plot(TFbinsPlot, tuningCurvesTFPlot(indDecrToFast, :), 'r', 'LineWidth', 0.5)
        ciplot(tuningDecrToFastConf(1,:), tuningDecrToFastConf(2,:), TFbinsPlot, 'r', 0.3)
    else
        tuningDecrToFastConf = zeros(size(tuningDecrToFastConf,1), size(tuningDecrToFastConf,2));
    end

    if length(indIncrToFast)>=5|| length(indDecrToFast)>=5
        ylabel('Mean change in Z-score', 'FontSize', 16, 'FontName', 'Arial')
        xlabel('Mean TF, Hz', 'FontSize', 16, 'FontName', 'Arial')
        xlim([TFbinsPlot(1) TFbinsPlot(end)])
        ylim([min(min([tuningIncrToFastConf tuningDecrToFastConf])) max(max([tuningDecrToFastConf tuningIncrToFastConf]))])
        % ylim([-1.5 1.5])
        title(brRegOfIntr)
    end
     

end
end


%% TF sequence

pValThresh = 0.001;
    figure('units','normalized','outerposition',[0.2 0.2 0.6 0.6]);

for i = 1:length(BrainRegGroups)
brRegGroup = BrainRegGroups{i};
brRegGroup = leaveBrRegWithEnoughUnits({unitPerBrainReg.name}, brRegGroup, {unitPerBrainReg.pValues}, pValThresh, 10);


for j = 1:length(brRegGroup)
    subplot(3, 3, (i-1)*3+j)
    brRegOfIntr = brRegGroup{j};

    brRegInd = find(strcmp({unitPerBrainReg.name}, brRegOfIntr)==1);
    signUnitsInd = find(unitPerBrainReg(brRegInd).pValues<pValThresh);

    TFSeqIncrPeakVal = reshape([unitPerBrainReg(brRegInd).TFSeqIncrPeakVal(signUnitsInd, :)], [], length(delayBTWpulsesPlot));
    TFSeqIncrPeakVal(find(abs(TFSeqIncrPeakVal(:,1))<1), :) = [];
    
    indIncrToFast = find(TFSeqIncrPeakVal(:,1)>0);
    indDecrToFast = find(TFSeqIncrPeakVal(:,1)<0);
    TFSeqIncrPeakVal =  TFSeqIncrPeakVal./(((TFSeqIncrPeakVal(:,1))));
%     TFSeqIncrPeakVal =  TFSeqIncrPeakVal-(((TFSeqIncrPeakVal(:,1))));


    % figure('units','normalized','outerposition',[0.2 0.2 0.2 0.4]);
    
    hold on
    plot([min(delayBTWpulsesPlot) max(delayBTWpulsesPlot)], [1 1], 'k--')
    
        TFSeqIncrPeakValConf = [];
        if length(indIncrToFast)>=5
            [~, ~, TFSeqIncrPeakValConf, ~] = normfit(TFSeqIncrPeakVal(indIncrToFast,:));
            plot(delayBTWpulsesPlot, mean(TFSeqIncrPeakVal(indIncrToFast, :)), 'k', 'LineWidth', 2)
            ciplot(TFSeqIncrPeakValConf(1,:), TFSeqIncrPeakValConf(2,:), delayBTWpulsesPlot, 'k', 0.25)
%             plot([min(delayBTWpulsesPlot) max(delayBTWpulsesPlot)], [1 1], 'k--')
        end
        
        TFSeqDecrPeakValConf = [];
        if strcmp(brRegOfIntr, 'GPe')==1&&length(indDecrToFast)>=5
            [~, ~, TFSeqDecrPeakValConf, ~] = normfit(TFSeqIncrPeakVal(indDecrToFast,:));
            plot(delayBTWpulsesPlot, mean(TFSeqIncrPeakVal(indDecrToFast, :)), 'k', 'LineWidth', 2)
            ciplot(TFSeqDecrPeakValConf(1,:), TFSeqDecrPeakValConf(2,:), delayBTWpulsesPlot, 'k', 0.25)
%             plot([min(delayBTWpulsesPlot) max(delayBTWpulsesPlot)], [-1 -1], 'k--')
        end
        
%     if length(indIncrToFast)>=5|| length(indDecrToFast)>=5
%         ylabel('Mean change in Z-score')
%         xlabel('Mean TF, Hz')
%         xlim([delayBTWpulsesPlot(1) delayBTWpulsesPlot(end)])
%         ylim([min(min([TFSeqIncrPeakValConf(:) TFSeqDecrPeakValConf(:)])) ceil(max(max([TFSeqDecrPeakValConf(:) TFSeqIncrPeakValConf(:)]))]) ])
        axis([delayBTWpulsesPlot(1) delayBTWpulsesPlot(end) 0 3])
        title(brRegOfIntr)
%     end
     

end
end





%% show single TF tuning
ind = 8501;

colors = lines(6);
colors = [[0 0 0] ; colors];
    
tuningCurveTF = allUnitsSumm.TF(ind).tuningCurveTF(2:end-1);
tuningCurveTFconf = allUnitsSumm.TF(ind).tuningCurveTFconf(2:end-1,:);
TFbinsMean = allUnitsSumm.TF(ind).TFbinsMean(2:end-1);

TFbinAvgFr = allUnitsSumm.TF(ind).TFbinAvgFr(2:end-1,:);

x = TFbinsMean(1):0.01:TFbinsMean(end);
P = polyfit(TFbinsMean,tuningCurveTF,1);
linfit = P(1)*x+P(2);

figure('units','normalized','outerposition',[0.3 0.2 0.25 0.4]);
set(gca, 'LineWidth', 1);
hold on
% h=plot(x,linfit, '--k','LineWidth',1);

plot([TFbinsMean(1) TFbinsMean(end)], [0 0], 'k--', 'LineWidth',1)
plot([1.02 1.02], [-50 50], 'k--', 'LineWidth',1)

plot(TFbinsMean, tuningCurveTF, 'k', 'LineWidth',2)
scatter(TFbinsMean(1), tuningCurveTF(1), 150, colors(4,:), 'filled')
scatter(TFbinsMean(2), tuningCurveTF(2), 150, colors(3,:), 'filled')
scatter(TFbinsMean(end-1), tuningCurveTF(end-1), 150, colors(2,:), 'filled')
scatter(TFbinsMean(end), tuningCurveTF(end), 150, colors(1,:), 'filled')

ciplot(tuningCurveTFconf(:,1), tuningCurveTFconf(:,2), TFbinsMean, 'k', 0.25)
corrTmp = corrcoef(TFbinsMean,tuningCurveTF);

% legend(h, ['R^2 = ' num2str(round(corr(1,2)^2*100)/100)], 'FontSize', 15)
% legend box off
ylabel('Change in Z-score', 'FontSize', 16)
xlabel('Mean TF, Hz', 'FontSize', 16)
yticks(-100:5:100)
xlim([TFbinsMean(1) TFbinsMean(end)])
ylim([min(min(tuningCurveTFconf([1 end],:))) max(max(tuningCurveTFconf([1 end],:)))]);
yl = ylim;
ylim([-max(abs(yl)) max(abs(yl))])

%%
figure('units','normalized','outerposition',[0.3 0.2 0.25 0.4]);
hold on
plot(PSTHwindow(1):binSize:PSTHwindow(end)-binSize, TFbinAvgFr(end,:)+TFbinAvgFr(end-1,:)-TFbinAvgFr(1,:)-TFbinAvgFr(2,:), 'k')
xlim([0 PSTHwindow(2)])
ylabel('Firing rate, Hz')
xlabel('Time, s')

end












