function Video = loadSessionVideo(sessionPathProcessed)

cd(sessionPathProcessed)
cd('Cameras')

motionEnergyFile = dir('motionEnergy.mat');
Video = load(fullfile(motionEnergyFile.folder, motionEnergyFile.name));

end

