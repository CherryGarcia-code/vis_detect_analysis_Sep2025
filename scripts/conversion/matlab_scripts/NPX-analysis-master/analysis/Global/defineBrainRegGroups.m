function [BrainRegGroups, BrainRegGroupNames] = defineBrainRegGroups

cortexVis = {'VISp', 'VISam', 'VISa', 'VISl', 'VISpl', 'VISrl', 'RSP', 'RSPd',  'RSPv'};
cortexRostAndMot = {'ACA', 'PL', 'ORB', 'ILA', 'MOs', 'MOp', 'AI', 'FRP'};
thalamus = {'LGd', 'LP', 'LD', 'VAL', 'VPL', 'VPM', 'RT', 'AV', 'CL', 'PO', 'PF', 'Eth', 'MD', 'MGv'};
cerebellum = {'SIM', 'CUL4 5', 'ANcr1', 'CENT', 'FN', 'IP', 'DN', 'PFL', 'FL', 'ANcr2'};
BG = {'LS', 'CP', 'GPe', 'GPi', 'SNr', 'PAL'};
midbrain = {'SCsl', 'SCiml', 'APN', 'MRN', 'MB', 'IC', 'NPC', 'RN', 'CUN'};
hippocampus = {'CA1', 'CA3', 'DG', 'SUB', 'POST', 'ProS', 'ENTm'};
hyp = {'LHA', 'ZI'};
ponsAndMedula = {'GRN', 'PB', 'IRN', 'NLL', 'SPVI', 'SPVO', 'V', 'VII'};

BrainRegGroups = {cortexVis, midbrain, thalamus, cortexRostAndMot, BG, cerebellum, ponsAndMedula, hippocampus, hyp};
BrainRegGroupNames = {'cortexVis',  'midbrain', 'thalamus','cortexRostAndMot', 'BG', 'cerebellum','ponsAndMedula', 'hippocampus', 'hyp'};

% BrainRegGroups = {cerebellum, cortexRostAndMot, thalamus, BG, cerebellum, midbrain, ponsAndMedula, hippocampus, hyp};

end