import os
import pickle
from PIL import Image
import numpy as np

tiny_imagenet_dir = "datasets/tiny-imagenet-200/"

# train & val splits
original_train_dir = os.path.join(tiny_imagenet_dir, "train")
original_train_subdirs = os.listdir(original_train_dir)
original_train_num = 500
actual_train_image_list = []
actual_train_target_list = []

# test split
original_val_dir = os.path.join(tiny_imagenet_dir, "val")
original_val_num = 50
actual_test_image_list = []
actual_test_target_list = []

# create  0 - 199 class index
idx_to_class = sorted(original_train_subdirs)
class_num = len(idx_to_class)

for class_idx in range(class_num):
    subdir = idx_to_class[class_idx]

    # train & val data
    original_train_subdir = os.path.join(original_train_dir, subdir)
    for image_idx in range(original_train_num):
        image_name = subdir + "_" + str(image_idx) + ".JPEG"
        original_train_image = os.path.join(original_train_subdir, image_name)
        assert (os.path.isfile(original_train_image))

        actual_train_image_list.append(np.asarray(Image.open(original_train_image).convert("RGB")))
        actual_train_target_list.append(class_idx)

    # test data
    original_val_subdir = os.path.join(original_val_dir, subdir)
    original_val_image_list = os.listdir(original_val_subdir)
    for original_val_image in original_val_image_list:
        actual_test_image_list.append(np.asarray(Image.open(os.path.join(original_val_subdir, original_val_image)).convert("RGB")))
        actual_test_target_list.append(class_idx)

assert (len(actual_train_image_list) == original_train_num * class_num)
assert (len(actual_test_image_list) == original_val_num * class_num)

actual_train_image_list = np.asarray(actual_train_image_list).transpose(0, 3, 1, 2)
actual_train_target_list = np.asarray(actual_train_target_list)
actual_test_image_list = np.asarray(actual_test_image_list).transpose(0, 3, 1, 2)
actual_test_target_list = np.asarray(actual_test_target_list)

pickle.dump(actual_train_image_list, open(os.path.join(tiny_imagenet_dir, "train_images.pkl"), "wb"))
pickle.dump(actual_train_target_list, open(os.path.join(tiny_imagenet_dir, "train_targets.pkl"), "wb"))
pickle.dump(actual_test_image_list, open(os.path.join(tiny_imagenet_dir, "test_images.pkl"), "wb"))
pickle.dump(actual_test_target_list, open(os.path.join(tiny_imagenet_dir, "test_targets.pkl"), "wb"))

print("Finished!")
