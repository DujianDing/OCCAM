from torch.utils.data import Dataset, DataLoader
import os
import pickle
from PIL import Image
from torchvision import transforms
import numpy as np


class Tiny_ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data = pickle.load(open(os.path.join(self.root, "{}_images.pkl".format(self.split)), "rb"))
        self.targets = pickle.load(open(os.path.join(self.root, "{}_targets.pkl".format(self.split)), "rb"))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_tinyimagenet(dataset_path, split, img_size=224):
    transform_tinyimagenet = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = Tiny_ImageNet(dataset_path, split=split, transform=transform_tinyimagenet)

    return dataset


if __name__ == "__main__":
    data_set = get_tinyimagenet("datasets/tiny-imagenet-200/", "train")
    # data_set = get_tinyimagenet("datasets/tiny-imagenet-200/", "val")
    # data_set = get_tinyimagenet("datasets/tiny-imagenet-200/", "test")
    data_loader = DataLoader(data_set, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    for _, data in enumerate(data_loader):
        image, target = data
