# Endo-STTN for Endoscopic Video Inpainting
![teaser](./docs/motivation.png?raw=true)

### [Paper](https://arxiv.org/abs/2203.17013) | [BibTex](#citation)

A Temporal Learning Approach to Inpainting Endoscopic Specularities and Its Effect on Image Correspondence<br>

Rema Daher, Francisco Vasconcelos, and Danail Stoyanov <br>
_Medical Image Analysis Journal 2023_.

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
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang},
  title = {Learning Joint Spatial-Temporal Transformations for Video Inpainting},
  booktitle = {The Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020}
}

```

<!-- ---------------------------------------------- -->
## Paper Contributions 
* A novel solution to specular highlight removal in endoscopic videos. This is achieved with a temporal learning based method.
* The generation of pseudo ground truth data that enables effective unsupervised training as well as quantitative evaluation.
* A quantitative and qualitative comparison of our approach
* Analysing qualitatively and quantitatively the effect of inpainting specular highlights on the estimation of stereo disparity, optical flow, feature matching, and camera motion

![Flowchart](./docs/Flowchart.png?raw=true)
![Architecture](./docs/FlowchartArchi.png?raw=true)


<!-- ---------------------------------------------- -->
## Installation  


```
git clone https://github.com/endomapper/Endo-STTN.git
cd Endo-STTN/
pip install -r requirements.txt
```

To install Pytorch, please refer to [Pytorch](https://pytorch.org/).

<!-- ---------------------------------------------- -->
## Dataset Preparation

Navigate to [./dataset_prep/README.md](./dataset_prep/README.md) for more details.

<!-- ---------------------------------------------- -->
## Inpaint Videos Using Pretrained Model

The inpainted frames can be generated using pretrained models. 
For your reference, we provide our model in ([pretrained model](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabrd0_ucl_ac_uk/ErDBwVttNuxKkWXG7nLsnQcBMxCrbWaRhpUBGEEQ_JE_ew?e=Nj4vwD)):
- Download and unzip the file in ./release_model/



### Testing script:


1. Arguments that can be set with [test.py](./test.py):
    - --overlaid: used to overlay the original frame pixels outside the mask region on your output. 
    - --shifted: used to inpaint using a shifted mask.
    - --framelimit: used to set the maximum number of frames per video (Default = 927).
    - --Dil: used to set the size of the structuring element used for dilation (Default = 8). If set to 0, no dilation will be made.

    >>**_NOTE_**: The testing script for shifted masked used for the paper differs slightly in the processing of the masks. For similar results like the paper use the following [masks](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabrd0_ucl_ac_uk/ElxeqDa9yVxKuNmgmnB3jSoB09sn6AgKQ2GRJfIgZtvzVQ?e=NBbJiQ) already shifted instead of --shifted. Similarly, [quantifyResults.ipynb](./quantifyResults.ipynb) should also be eddited to use these masks and  _shifted_ should be set to False.

<br />

2. To test on all the test videos in your dataset, listed in your test.json (refer to [./dataset_prep/README.md](./dataset_prep/README.md)):
    ```
    python test.py --gpu <<INSERT GPU INDEX>> --overlaid \
    --output <<INSERT OUTPUT DIR>> \
    --frame <<INSERT FRAMES DIR>> \ # Folder should be called JPEGImages
    --mask <<INSERT ANNOTATIONS DIR>> \
    -c <<INSERT PRETRAINED PARENT DIR>> \
    -cn <<INSERT PRETRAINED MODEL NUMBER>> \
    --zip
    ``` 

    - For example, using pretrained model release_model/pretrained_model/gen_00009.pth: 
      ```
      python test.py --gpu 1 --overlaid \
      --output results/Inpainted_pretrained/ \
      --frame datasets/EndoSTTN_dataset/JPEGImages/ \
      --mask datasets/EndoSTTN_dataset/Annotations/ \
      -c release_model/pretrained_model/ \
      -cn 9 \
      --zip
      ```
    >>**_NOTE_**: When running this script the loaded frames and masks are saved as npy files in datasets/EndoSTTN_dataset/files/so that loading them would be easier if you want to rerun this script. To load these npy files use the --readfiles argument. This is useful when experimenting with a large dataset.

3. To test on 1 video: 
    ```
    python test.py --gpu <<INSERT GPU INDEX>> --overlaid \
    --output <<INSERT VIDEO OUTPUT DIR>> \
    --frame <<INSERT VIDEO FRAMES DIR>> \
    --mask <<INSERT VIDEO ANNOTATIONS DIR>> \
    -c <<INSERT PRETRAINED PARENT DIR>> \
    -cn <<INSERT PRETRAINED MODEL NUMBER>>
    ``` 

    - For example, for a folder "ExampleVideo1_Frames" containing the video frames, using pretrained model release_model/pretrained_model/gen_00009.pth: 

      ``` 
      python test.py  --gpu 1 --overlaid \
      --output results/Inpainted_pretrained/ \
      --frame datasets/ExampleVideo1_Frames/ \
      --mask datasets/ExampleVideo1_Annotations/ \
      -c release_model/pretrained_model/ \
      -cn 9
      ``` 

4. Single frame testing:

    To test a single frame at a time and thus removing the temporal component, follow the same steps above but use [test-singleframe.py](./test-singleframe.py) instead of [test.py](./test.py).


<!-- ---------------------------------------------- -->
## Training New Models
Once the dataset is ready, new models can be trained:
- Prepare the configuration file (ex: [EndoSTTN_dataset.json](./configs/EndoSTTN_dataset.json)):
  - "gpu": \<INSERT GPU INDICES EX: "1,2"\>
  - "data_root": \<INSERT DATASET ROOT\>
  - "name": \<INSERT NAME OF DATASET FOLDER\>
  - "frame_limit": used to set the maximum number of frames per video (Default = 927). In the paper no limit was used for training; only for testing.
  - "Dil": used to set the size of the structuring element used for dilation (Default = 8). If set to 0, no dilation will be made.

 <br />

### Training Script

```
python train.py --model sttn \
--config <<INSERT CONFIG FILE DIR>> \
-c <<INSERT INITIALIZATION MODEL PARENT DIR>> \
-cn <<INSERT INITIALIZATION MODEL NUMBER>>
```
- For example, using pretrained model release_model/pretrained_model/gen_00009.pth as initialization: 
  ```
  python train.py --model sttn \
  --config configs/EndoSTTN_dataset.json \
  -c release_model/pretrained_model/ \
  -cn 9
  ```


<!-- ---------------------------------------------- -->
## Evaluation 

>>**_NOTE_**: Refer to comment in [Testing Script (1.)](#testing-script).

To quantitatively evaluate results using pseudo-ground-truth:
1. Test all videos using [Testing Script (2.)](#testing-script) with the **--shifted** argument.
2. Install Scikit Image: ``` pip install scikit-image==0.15.0 ```.
3. Use [quantifyResults.ipynb](./quantifyResults.ipynb) to generate csv files containing the quantitative results.




<!-- ---------------------------------------------- -->
## Visualization 

We provide an example of visualization attention maps in ```visualization.ipynb```. 


<!-- ---------------------------------------------- -->
## Training Monitoring  

We provide training monitoring on losses by running: 
```
tensorboard --logdir release_mode                                                    
```

<!-- ---------------------------------------------- -->
## Contact
If you have any questions or suggestions about this paper, feel free to contact me (remadaher711@gmail.com).