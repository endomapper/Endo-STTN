# Dataset Preparation


Our dataset is based on the Hyper Kvasir dataset, please also cite it if used:
```
@article{borgli2020hyperkvasir,
  title={HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy},
  author={Borgli, Hanna and Thambawita, Vajira and Smedsrud, Pia H and Hicks, Steven and Jha, Debesh and Eskeland, Sigrun L and Randel, Kristin Ranheim and Pogorelov, Konstantin and Lux, Mathias and Nguyen, Duc Tien Dang and others},
  journal={Scientific data},
  volume={7},
  number={1},
  pages={1--14},
  year={2020},
  publisher={Nature Publishing Group}
}
```

1. To download our dataset:
    - Download it from ([Dataset](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabrd0_ucl_ac_uk/EoyhTw5vdQBHr8-r-Iv-XfcB5E_88GkMuEddnRVKxwfQKQ?e=XBY9Dg))
    - Unzip it in [datasets](./datasets/)
    - It should look like this:

    ```
    datasets
      |- EndoSTTN_dataset
            |- JPEGImages
              |- <video_id>.zip
              |- <video_id>.zip
            |- Annotations
              |- <video_id>.zip
              |- <video_id>.zip        
            |- test.json 
            |- train.json 
    ``` 
    train.json includes video IDs to be used for training with their number of frames:
    {"video_id": 1982,
    "video_id": 3509,
    "video_id": 1569}

    test.json includes video IDs to be used for testing with their number of frames:
    {"video_id": 1982,
    "video_id": 3509,
    "video_id": 1569}                 

2. To create your dataset from scratch use this Hyper Kvasir example:
    ```
    cd dataset_prep
    ```

    Download hyper-kvasir videos:
    ```
    wget -O hyper-kvasir-labeled-videos.zip "https://files.de-1.osf.io/v1/resources/mh9sj/providers/osfstorage/5dfb57eaa95d73000939fc5b/?zip=labeled-videos.zip"
    mkdir labeled-videos
    cd labeled-videos
    unzip ../hyper-kvasir-labeled-videos.zip
    rm ../hyper-kvasir-labeled-videos.zip
    cd ..
    ```
    Now that you have the hyper kvasir videos, run the following:
    ```
    python rename.py
    python video2frames.py
    python resizeFrames288.py
    python folder_dicts.py
    ```
    - From "train.json", move the video IDs that are selected for testing with their frame count to a similar file "test.json".
    
      - train.json includes video IDs to be used for training with their number of frames:
    {"video_id": 1982,
    "video_id": 3509,
    "video_id": 1569}

      - test.json includes video IDs to be used for testing with their number of frames:
    {"video_id": 1982,
    "video_id": 3509,
    "video_id": 1569} 
    - To get the specular highlight masks we rely on the following code ([Github Repo](https://github.com/jiemojiemo/some_specular_detection_and_inpainting_methods_for_endoscope_image/tree/master/Automatic_detection_and_Inpainting_of_specular_reflections_Othmane_2011)):
      - In Matlab run ([createSpecularMasks288.m](./Automatic_detection_and_Inpainting_of_specular_reflections_Othmane_2011/createSpecularMasks288.m))
    - zip folders:
    ```
    cd ./labeled-videos-Processed/Resized/Frames/
    bash ../../../zipper.sh 
    mkdir ../FramesZipped/
    mv *.zip ../FramesZipped/
    cd ../Annotations/
    bash ../../../zipper.sh 
    mkdir ../AnnotationsZipped/
    mv *.zip ../AnnotationsZipped/
    mv ../../../*json ../
    ```

    - Move and rename FramesZipped, AnnotationsZipped, test.json, and train.json in labeled-videos-Processed/Resized/ to the datasets folder to look like this:
    ```
    datasets
      |- EndoSTTN_dataset
            |- JPEGImages
              |- <video_id>.zip
              |- <video_id>.zip
            |- Annotations
              |- <video_id>.zip
              |- <video_id>.zip        
            |- test.json 
            |- train.json 
    ``` 

