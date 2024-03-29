# VHH Plugin Package: Relation Detection
This package includes methods to compute similarity between scenes and find scenes that are similar to a given query scene.

## Package Description
Will be added later.
    

## Quick Setup

The following instructions have to be done to used this library in your own application:

**Requirements:**

   * Ubuntu 18.04 LTS
   * CUDA 10.1 + cuDNN
   * python version 3.6.x

We developed and tested this module with pytorch 1.10.0+cu113 and torchvision 0.11.1+cu113.
   
### 0 Environment Setup (optional)

**Create a virtual environment:**

   * ```create a folder to a specified path (e.g. /xxx/vhh_rd_env/)```
   * ```python3 -m venv /xxx/vhh_rd_env/```

**Activate the environment:**

   * ```source /xxx/vhh_rd_env/bin/activate```

### 1 Install directly from GitHub

**Clone the repository:**

   * ```git clone https://github.com/dahe-cvl/vhh_rd```

**Install requirements:**

   * ```pip install -r requirements.txt```

**Set paths in config file:**

  * Open the config file at ```config/config_rd.yaml```
  * Set ```VIDEO_PATH``` to the directory containing the video you want to analyze
  * Set ```SHOT_PATH``` to the directory containing the shot information generated with [VHH_SBD](https://github.com/dahe-cvl/vhh_sbd) 
  * Set ```DATA_PATH``` to the directoy which you want to store all the data generated by VHH_RD

### 2 Run the scripts (optional)

  * To download all annotated films, run ```python download_videos.py -p /data/ext/VHH/datasets/vhh_mmsi_v1_5_0_relation_db/films -a```
  * To download all sbd annotations as csv, run ```python download_annotation_results.py -a -c -p /data/ext/VHH/datasets/vhh_mmsi_v1_5_0_relation_db/annotations```
  * To extract center frames from shots and compute feature vectors, run ```python Demo/run_vhh_rd.py```
  * To visualize the computed feature vectors, run ```python Demo/visualize_dataset.py```
  * To find similar frames for a frame with the name $IMG_NAME in the ```ExtractedFrames``` directory, run ```python Demo/find_similar_images.py -i IMG_NAME```

## Release Generation
Will be added later.
