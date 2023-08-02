clc;
clear all;
close all;
%dbstop if error

%ADD FILEPATH HERE
D=dir('./labeled-videos-Processed/Resized/Frames/hyperK*/*.jpg');

for j = 1:length(D)
	img_path=fullfile(D(j).folder,D(j).name);
    img = imread(img_path);
    specular_mask = SpecularDetectionMeslouhi2011(img);
    %dilated_mask = imdilate(specular_mask, strel('diamond', 1));
    %file path for annotations
   	[folder_temp, foldername, ~] = fileparts(D(j).folder);
    [Anot, ~, ~] = fileparts(folder_temp);
    if ~exist(fullfile(Anot,'Annotations',foldername),'dir')
        mkdir(fullfile(Anot,'Annotations',foldername));
    end
    Anotspec=fullfile(Anot,'Annotations',foldername, D(j).name);
    %write files
    imwrite(specular_mask,Anotspec)
    %imwrite(dilated_mask,Anotdil)
end

