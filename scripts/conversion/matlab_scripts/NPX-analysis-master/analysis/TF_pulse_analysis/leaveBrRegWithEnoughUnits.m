function brRegGroup = leaveBrRegWithEnoughUnits(allBrainRegNames, brRegGroup, pValues, pValThresh, minSignUnitNumb)

showBrReg = zeros(1, length(brRegGroup));
for i = 1:length(brRegGroup)
    brRegOfIntr = brRegGroup{i};
    brRegInd = find(strcmp(allBrainRegNames, brRegOfIntr)==1);
    if ~isempty(brRegInd)
        signUnitsInd = find(pValues{brRegInd}<pValThresh);
        if length(signUnitsInd)>=minSignUnitNumb
            showBrReg(i) = 1;
        end
    end        
end
brRegGroup(showBrReg==0) = [];

end

