import os
from shutil import copyfile

# split GTA5
data_path = "working_directory/data/GTA5"
sub_dir = "images"
data_list = 'gta5_list'

# split SYNTHIA
# data_path = "working_directory/data/SYNTHIA"
# sub_dir = "RGB"
# data_list = 'synthia_list'

train_list_path = os.path.join('.', data_list, 'train.txt')
train_list = [i_id for i_id in open(train_list_path)]
for train_file in train_list:
    train_file = train_file[:-1]
    file_path_train = os.path.join(data_path, sub_dir, train_file)
    if not os.path.exists(os.path.join(data_path, "train")):
        os.makedirs(os.path.join(data_path, "train"))
    copyfile(file_path_train, os.path.join(data_path, "train", train_file))

val_list_path = os.path.join('.', data_list, 'val.txt')
val_list = [i_id for i_id in open(val_list_path)]
for val_file in val_list:
    val_file = val_file[:-1]
    file_path_val = os.path.join(data_path, sub_dir, val_file)
    if not os.path.exists(os.path.join(data_path, "val")):
        os.makedirs(os.path.join(data_path, "val"))
    copyfile(file_path_val, os.path.join(data_path, "val", val_file))
