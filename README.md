 
    
# Dataset Download

- GTA5

  https://download.visinf.tu-darmstadt.de/data/from_games/

- SYNTHIA

  http://synthia-dataset.net/downloads/
  (SYNTHIA-RAND-CITYSCAPES (CVPR16))

- CityScapes

  https://www.cityscapes-dataset.com/

- IDD

  https://idd.insaan.iiit.ac.in/dataset/download/
  (IDD - Segmentation (IDD 20k Part I))
  https://drive.google.com/file/d/1UUI4_101KCw17hEdqJIA7DlBSk4Q7ndu/view?usp=sharing
  (Download the file from the link. Extract the file, and replace the gtFine folder in the IDD dataset with this one.)

- Cross-City

  https://yihsinchen.github.io/segmentation_adaptation/

Place four datasets in the ./data folder.

Change each folder's name as 'GTA5', 'SYNTHIA', 'CityScapes', 'IDD', 'NTHU_Datasets', respectively.


# Split Dataset

Run
```
./dataset/split_dataset.py
```
for GTA5 and SYNTHIA, respectively.


# Download Pre-trained Model

https://drive.google.com/file/d/16TRsmELvoLKJuaijRJF9NK_SJq3YagYw/view?usp=sharing

Place the file in the ./pretrained folder.


# Training

## For CityScapes and IDD
  Run 
  ```
  train.py
  ```
  The options should be changed before running.
  Change the options in ```options_train.py```

## For Cross-City
  Run 
  ```
  trainCrossCity.py
  ```
  The options should be changed before running.
  Change the options in ```options_trainCrossCity.py```


# Evaluation

## For CityScapes and IDD
  Run 
  ```
  evaluate.py
  ```
  The options should be changed before running.
  The options are contained in the ```evaluate.py``` file.


## For Cross-City
  Run 
  ```
  evaluateCrossCity.py
  ```
  The options should be changed before running.
  The options are contained in the ```evaluateCrossCity.py``` file.

*** This codebase is heavily borrowed from a source code of AdaptSegNet. (https://github.com/wasidennis/AdaptSegNet) ***
