1. Dataset Download

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
(Download the file from the above link. Extract the file, and replace the gtFine folder in the IDD dataset with this one.)

Place four datasets in the ./data folder.

Change each folder's name as 'GTA5', 'SYNTHIA', 'CityScapes', 'IDD', respectively.


2. Split Dataset

Run ./dataset/split_dataset.py for GTA5 and SYNTHIA, respectively.


3. Training

Run train.py

The options should be changed before running.
Change the options by options_train.py


4. Evaluation

Run evaluate.py

The options should be changed before running.
The options are contained in the evaluate.py file.


* This codebase is heavily borrowed from a source code of AdaptSegNet.
