function TFbins = makeTFbins(TFdistr, eventsFracPerBin, TFmin, TFmax)
%MAKETFBINS Summary of this function goes here
%   Detailed explanation goes here

eventsNumbPerBin = round( length(TFdistr)*eventsFracPerBin);
TFbinStart = TFmin;
TFbinEnd = TFmin;
TFdistr = sort(TFdistr);
i = 1;
while TFbinEnd<=TFmax
    if i ==1
        TFbins(i,1) = TFbinStart;
    else
        TFbins(i,1) = TFbinStartNew;
    end

    indStart = find(TFdistr>=TFbins(i,1), 1, 'first');
    try
        TFbinEnd = TFdistr(indStart + eventsNumbPerBin);
        TFbins(i,2) = TFbinEnd;
%         TFbinStartNew = mean(TFdistr(indStart:indStart + eventsNumbPerBin));   % use the middle of the current bin as a start for next bin
        TFbinStartNew = (TFdistr(indStart + eventsNumbPerBin));   

        i = i+1;
    catch
        TFbins(i,:) = [];
        TFbinEnd = 1.1*TFmax;
    end
end

