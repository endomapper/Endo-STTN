clc;
clear all;
close all;
dbstop if error

cd '/home/rema/STTN/Automatic_detection_and_Inpainting_of_specular_reflections_Othmane_2011/'
addpath ./lib

D = dir('/home/rema/STTN/datasets/hyper-kvasir/JPEGImages/**/*.jpg');

for k = 1:height(D)
    img_path=fullfile(D(k).folder,D(k).name);
    img = imread(img_path);
    specular_mask = SpecularDetectionMeslouhi2011(img);
    dilated_mask = imdilate(specular_mask, strel("diamond", 1));
    %file path for annotations 
    Anot=split(img_path,'JPEGImages');
    Anotspec=fullfile(Anot{1},'AnnotationsSpec',Anot{2});
    Anotdil=fullfile(Anot{1},'AnnotationsDil',Anot{2});
    %write files
    imwrite(specular_mask,Anotspec)
    imwrite(dilated_mask,Anotdil)    
end
