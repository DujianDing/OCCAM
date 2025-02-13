import torch
from torch.utils.data import DataLoader
import os
from torchvision import transforms, datasets, models
from distutils.version import LooseVersion
from torch.utils.tensorboard import SummaryWriter
from dataset import get_datasets
from classifier import get_classifier
import numpy as np
import datetime
import pickle
import argparse


def main(args):
    # parse args
    dataset_name = args["dataset_name"]
    dataset_path = args["dataset_path"]
    network_name = args["network_name"]
    network_path = args["network_path"]
    classifier_path = args["classifier_path"]
    r_seed = args["r_seed"]
    batch_size = args["batch_size"]
    num_workers = args["num_workers"]
    version_suffix = args["version_suffix"]

    # set random seed
    torch.manual_seed(r_seed)
    np.random.seed(r_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(dataset_path): os.makedirs(dataset_path)
    if not os.path.exists(network_path): os.makedirs(network_path)
    if not os.path.exists(classifier_path): os.makedirs(classifier_path)
    torch.hub.set_dir(network_path)
    host_device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 224

    if network_name == "resnet18":
        nn_classifier = models.resnet18(weights='IMAGENET1K_V1')
    elif network_name == "resnet34":
        nn_classifier = models.resnet34(weights='IMAGENET1K_V1')
    elif network_name == "resnet50":
        nn_classifier = models.resnet50(weights='IMAGENET1K_V1')
    elif network_name == "resnet101":
        nn_classifier = models.resnet101(weights='IMAGENET1K_V1')
    elif network_name == "resnet152":
        nn_classifier = models.resnet152(weights='IMAGENET1K_V1')
    elif network_name == "swin_v2_t":
        nn_classifier = models.swin_v2_t(weights="IMAGENET1K_V1")
    elif network_name == "swin_v2_s":
        nn_classifier = models.swin_v2_s(weights="IMAGENET1K_V1")
    elif network_name == "swin_v2_b":
        nn_classifier = models.swin_v2_b(weights="IMAGENET1K_V1")
    elif network_name == "vit_l_16":
        nn_classifier = models.vit_l_16(weights="IMAGENET1K_SWAG_LINEAR_V1")
    elif network_name == "vit_h_14":
        nn_classifier = models.vit_h_14(weights="IMAGENET1K_SWAG_LINEAR_V1")
    else:
        raise NotImplementedError(network_name)
    nn_classifier = nn_classifier.to(host_device)
    nn_classifier.eval()

    # get data sets and data loaders
    transform_imagenet = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None,
                          antialias=False),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_set = datasets.ImageNet(root=dataset_path, split='val', transform=transform_imagenet)
    val_number = int(len(test_set) * 0.4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    # random shuffle test set
    if os.path.exists(dataset_path + f"validation_test_indices_{val_number}.pkl"):
        print(f"Loading validation_test_indices_{val_number}.pkl...")
        test_idxs, val_idxs = pickle.load(open(dataset_path + f"validation_test_indices_{val_number}.pkl", "rb"))
    else:
        print(f"Generating validation_test_indices_{val_number}.pkl...")
        shuffled_val_test_idxs = np.random.permutation(len(test_loader))
        val_idxs = shuffled_val_test_idxs[:val_number]
        test_idxs = shuffled_val_test_idxs[val_number:]
        pickle.dump((test_idxs, val_idxs), open(dataset_path + f"validation_test_indices_{val_number}.pkl", "wb"))

    # test set inference
    x_target_list = []
    x_logit_list = []
    for _, data in enumerate(test_loader):
        image, target = data
        x_feature = image.to(host_device)
        x_logit = nn_classifier(x_feature).detach().cpu().numpy()

        x_target_list.append(target)
        x_logit_list.append(x_logit)

    x_target_list = np.asarray(x_target_list)
    x_logit_list = np.asarray(x_logit_list)
    # save test set inference results
    torch.save(x_target_list[test_idxs], classifier_path + "/test_target_list.pt")
    torch.save(x_logit_list[test_idxs], classifier_path + "/test_logit_list.pt")
    torch.save(x_target_list[val_idxs], classifier_path + "/val_target_list.pt")
    torch.save(x_logit_list[val_idxs], classifier_path + "/val_logit_list.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, type=str, help="network name")
    parser.add_argument("--dataset", required=True, type=str, help="dataset name")
    in_args = parser.parse_args()

    network_name = in_args.network
    if network_name not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                            "swin_v2_t", "swin_v2_s", "swin_v2_b", "vit_l_16", "vit_h_14"]:
        raise NotImplementedError(network_name)

    dataset_name = in_args.dataset
    if dataset_name not in ["imagenet"]:
        raise NotImplementedError(dataset_name)

    args = {
        "dataset_name": dataset_name,
        "dataset_path": f"datasets/{dataset_name}/",
        "network_name": network_name,
        "network_path": f"networks/{network_name}/",
        "classifier_path": os.path.join("classifiers", dataset_name,
                                        network_name + '_' +
                                        datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")),
        "r_seed": 0,
        "batch_size": 1,
        "num_workers": 0,
        "version_suffix": f"_{network_name}",
    }
    main(args)
