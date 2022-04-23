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
## Completing Videos Using Pretrained Model

The inpainted frames can be generated using pretrained models. 
For your reference, we provide our model in ([pretrained model](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabrd0_ucl_ac_uk/ERzrr0GgtLVGrJhxLvYS6_ABt19Iva2d0x7ijouWyo1Vog?e=IQwlJf)). 

Use a similar pipeline to [Dataset Preparation](#dataset-preparation) to create the following folders with only the testing sequences:
  ```
  datasets
    |- EndoSTTN_dataset-Testing
          |- JPEGImages-Testing
            |- <video_id>.zip
            |- <video_id>.zip
          |- Annotations-Testing
            |- <video_id>.zip
            |- <video_id>.zip        
  ``` 
    
Complete frames using the pretrained model. For example, 

```
python test.py --output <<INSERT OUTPUT DIR>> --frame <<INSERT DIR OF JPEGImages>> --mask <<INSERT DIR OF Annotations>> --ckpt pretrained/gen_00009.pth
python test.py --output datasets/EndoSTTN_dataset-Testing/Inpaintedframes/ --frame datasets/EndoSTTN_dataset-Testing/JPEGImages-Testing/ --mask Annotations-Testing/ --ckpt pretrained/gen_00009.pth
``` 


<!-- ---------------------------------------------- -->
## Dataset Preparation
To use our dataset, either:
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
    train.json includes video ids to be used for training with their number of frames:
    {"video_id": 1982,
    "video_id": 3509,
    "video_id": 1569}

    test.json includes video ids to be used for testing with their number of frames:
    {"video_id": 1982,
    "video_id": 3509,
    "video_id": 1569}                 

2. To create dataset from scratch:
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
    ```
    - To get the specular highlight masks we rely on the code of ([Github Repo](https://github.com/jiemojiemo/some_specular_detection_and_inpainting_methods_for_endoscope_image/tree/master/Automatic_detection_and_Inpainting_of_specular_reflections_Othmane_2011)):
      - In Matlab run ([createSpecularMasks288.m](./Automatic_detection_and_Inpainting_of_specular_reflections_Othmane_2011/))
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
    train.json includes video ids to be used for training with their number of frames:
    {"video_id": 1982,
    "video_id": 3509,
    "video_id": 1569}

    test.json includes video ids to be used for testing with their number of frames:
    {"video_id": 1982,
    "video_id": 3509,
    "video_id": 1569} 


<!-- ---------------------------------------------- -->
## Training New Models
Once the dataset is ready, new models can be trained:
- Remove this in train.py if not needed (used to choose only one GPU):    ```os.environ["CUDA_VISIBLE_DEVICES"]="7"``` 
- In the config file:
  - "data_root": \<INSERT DATASET ROOT\>
  - "name": \<INSERT NAME OF DATASET FOLDER\>
- Use --initialmodel to initialize your model with a pretrained one (in this example the pretrained model is called sttn_EndoSTTN_dataset)

```
python train.py --config configs/EndoSTTN_dataset.json --model sttn --initialmodel release_model/sttn_EndoSTTN_dataset/
```


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
