matlab

clc;
clear all;
close all;
%dbstop if error

cd './Automatic_detection_and_Inpainting_of_specular_reflections_Othmane_2011/'
addpath ./lib

%ADD FILEPATH HERE
D=dir('./labeled-videos-Processed/Resized/Frames/hyperK*/')

for j = 1:length(D)
    folderD=fullfile('./labeled-videos-Processed/Resized/Frames/',D(j).name)
    D2=dir(fullfile(folderD,'*.jpg'))
    for k = 1:length(D2)
	img_path=fullfile(folderD,D2(k).name);
    	img = imread(img_path);
    	specular_mask = SpecularDetectionMeslouhi2011(img);
    	dilated_mask = imdilate(specular_mask, strel('diamond', 1));
    	%file path for annotations
   	[folder_temp, foldername, ~] = fileparts(folderD);
        [Anot, ~, ~] = fileparts(folder_temp);
    	mkdir(fullfile(Anot,'Annotations',foldername))
    	Anotdil=fullfile(Anot,'Annotations',foldername, D2(k).name);
    	%write files
    	%imwrite(specular_mask,Anotspec)
    	imwrite(dilated_mask,Anotdil)
     end
end

