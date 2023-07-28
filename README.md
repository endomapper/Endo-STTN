# Endo-STTN for Endoscopic Video Inpainting
![teaser](./docs/motivation.png?raw=true)

### [Paper](https://arxiv.org/abs/2203.17013) | [BibTex](#citation)

A Temporal Learning Approach to Inpainting Endoscopic Specularities and Its effect on Image Correspondence<br>

Rema Daher, Francisco Vasconcelos, and Danail Stoyanov <br>
arXiv 2022.

<!-- ---------------------------------------------- -->
## Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@article{daher2022temporal,
  title={A Temporal Learning Approach to Inpainting Endoscopic Specularities and Its effect on Image Correspondence},
  author={Daher, Rema and Vasconcelos, Francisco and Stoyanov, Danail},
  journal={arXiv preprint arXiv:2203.17013},
  year={2022}
}
```

Since this code is based on [STTN](https://github.com/researchmm/STTN), please also cite their work: 
```
@inproceedings{yan2020sttn,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang,
  title = {Learning Joint Spatial-Temporal Transformations for Video Inpainting},
  booktitle = {The Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020}
}

```

<!-- ---------------------------------------------- -->
## Paper Contributions 
* A novel solution to specular highlight removal in endoscopic videos. This is achieved with a temporal learning based method.
* The generation of pseudo ground truth data that enables effective unsupervised training as well as quantitative evaluation
* A quantitative and qualitative comparison of our approach
* Analysing qualitatively and quantitatively the effect of inpainting specular highlights on the estimation of stereo disparity, optical flow, feature matching, and camera motion

![Flowchart](./docs/Flowchart.png?raw=true)
![Architecture](./docs/FlowchartArchi.png?raw=true)


<!-- ---------------------------------------------- -->
## Installation  

Clone this repo.

```
git clone https://github.com/endomapper/Endo-STTN.git
cd Endo-STTN/
```

We build our project based on Pytorch and Python. For the full set of required Python packages, we suggest creating a Conda environment from the provided YAML, e.g.

```
conda env create -f environment.yml 
conda activate sttn
```


<!-- ---------------------------------------------- -->
## Dataset Preparation
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
    - Place it in folder datasets
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
    jar -xvf ../hyper-kvasir-labeled-videos.zip
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
    - Run zipper.sh inside Frames and Annotations in labeled-videos-Processed/Resized/ to zip the video folders inside them.
    - Move and rename Frames and Annotations in labeled-videos-Processed/Resized/ to the datasets folder to look like this:
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


<!-- ---------------------------------------------- -->
## Inpaint Videos Using Pretrained Model

The inpainted frames can be generated using pretrained models. 
For your reference, we provide our model in ([pretrained model](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabrd0_ucl_ac_uk/ErDBwVttNuxKkWXG7nLsnQcBMxCrbWaRhpUBGEEQ_JE_ew?e=Nj4vwD)):
- Download and unzip the file in the release_model/ folder.

### Testing script:

1. Arguments that can be set with test.py:
    - --overlaid: used to overlay the original frame pixels outside the mask region on your output. 
    - --shifted to inpaint using a shifted mask.
    - --framelimit used to set the maximum number of frames per video (Default = 927).
    - --Dil used to set the size of structuring element used for dilation (Default = 8). If set to 0, no dilation will be made.

<br />

2. To test on all the videos listed in your test.json:
    ```
    python test.py --gpu <<INSERT GPU INDEX>> --overlaid \
    --output <<INSERT OUTPUT DIR>> \
    --frame <<INSERT JPEGImages DIR>> \
    --mask <<INSERT ANNOTATIONS DIR>> \
    --c <<INSERT PRETRAINED PARENT DIR>> \
    --cn <<INSERT PRETRAINED MODEL NUMBER>> \
    --zip
    ``` 

    - For example, using pretrained model release_model/pretrained_model/gen_00009.pth: 
      ```
      python test.py --gpu 1 --overlaid \
      --output results/Inpainted_pretrained_gen9/ \
      --frame datasets/EndoSTTN_dataset/JPEGImages/ \
      --mask datasets/EndoSTTN_dataset/Annotations/ \
      --c release_model/pretrained_model/ \
      --cn 9 \
      --zip
      ```

3. To test on 1 video: 
    ```
    python test.py --gpu <<INSERT GPU INDEX>> --overlaid \
    --output <<INSERT VIDEO OUTPUT DIR>> \
    --frame <<INSERT VIDEO FRAMES DIR>> \
    --mask <<INSERT VIDEO ANNOTATIONS DIR>> \
    --c <<INSERT PRETRAINED PARENT DIR>> \
    --cn <<INSERT PRETRAINED MODEL NUMBER>>
    ``` 

    - For example, for a folder "ExampleVideo1_Frames" containing the video frames, using pretrained model release_model/pretrained_model/gen_00009.pth: 

      ``` 
      python test.py  --gpu 1 --overlaid \
      --output results/Inpainted_pretrained_gen9/ExampleVideo1_Inpainted/ \
      --frame datasets/ExampleVideo1_Frames/ \
      --mask datasets/ExampleVideo1_Annotations/ \
      --c release_model/pretrained_model/ \
      --cn 9
      ``` 

4. Single frame testing:

    To test a single frame at a time and thus removing the temporal component, follow the same steps above but use **test-singleframe.py** instead of **test.py**.

<!-- ---------------------------------------------- -->
## Training New Models
Once the dataset is ready, new models can be trained:
- Prepare the configuration file (ex: [EndoSTTN_dataset.json](./configs/EndoSTTN_dataset.json)):
  - "gpu": \<INSERT GPU INDICES EX: "1,2"\>
  - "data_root": \<INSERT DATASET ROOT\>
  - "name": \<INSERT NAME OF DATASET FOLDER\>
  - "frame_limit": used to set the maximum number of frames per video (Default = 927).
  - "Dil": used to set the size of structuring element used for dilation (Default = 8). If set to 0, no dilation will be made.

 <br />

- train.py will be used for training:

  ```
  python test.py --gpu <<INSERT GPU INDEX>> --overlaid \
  --config <<INSERT CONFIG FILE DIR>> \
  --c <<INSERT INITIALIZATION MODEL PARENT DIR>> \
  --cn <<INSERT INITIALIZATION MODEL NUMBER>>
  ```
  - For example, using pretrained model release_model/pretrained_model/gen_00009.pth as initialization: 
    ```
    python train.py --model sttn \
    --config configs/EndoSTTN_dataset.json \
    --c release_model/pretrained_model/ \
    --cn 9
    ```


<!-- ---------------------------------------------- -->
## Evaluation 
To quantitatively evaluate results using pseudo-ground-truth:
1. Test all videos using [Testing Script (2.)](#testing-script) and adding the **--shifted** argument.
2. Use [quantifyResults.ipynb](./quantifyResults.ipynb) to generate csv files containing the quantitative results.


<!-- ---------------------------------------------- -->
## Visualization 

We provide an example of visualization attention maps in ```visualization.ipynb```. 


<!-- ---------------------------------------------- -->
## Training Monitoring  

We provide traning monitoring on losses by running: 
```
tensorboard --logdir release_mode                                                    
```

<!-- ---------------------------------------------- -->
## Contact
If you have any questions or suggestions about this paper, feel free to contact me (remadaher711@gmail.com).