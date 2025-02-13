import torch
from torchvision import transforms, datasets, models
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
import os
import gc
import numpy as np
import pickle
from collections import Counter
import argparse
from tinyimagenet_dataset import Tiny_ImageNet


def main(args):
    # parse args
    dataset_name = args["dataset_name"]
    dataset_path = args["dataset_path"]
    network_name = args["network_name"]
    network_path = args["network_path"]
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
    torch.hub.set_dir(network_path)
    host_device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 224

    if network_name == "resnet18":
        nn_model_resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        return_nodes = {
            # node_name: user-specified key for output dict
            'flatten': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_resnet18, return_nodes)
    elif network_name == "resnet34":
        nn_model_resnet34 = models.resnet34(weights='IMAGENET1K_V1')
        return_nodes = {
            # node_name: user-specified key for output dict
            'flatten': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_resnet34, return_nodes)
    elif network_name == "resnet50":
        nn_model_resnet50 = models.resnet50(weights='IMAGENET1K_V1')
        return_nodes = {
            # node_name: user-specified key for output dict
            'flatten': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_resnet50, return_nodes)
    elif network_name == "resnet101":
        nn_model_resnet101 = models.resnet101(weights='IMAGENET1K_V1')
        return_nodes = {
            # node_name: user-specified key for output dict
            'flatten': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_resnet101, return_nodes)
    elif network_name == "resnet152":
        nn_model_resnet152 = models.resnet152(weights='IMAGENET1K_V1')
        return_nodes = {
            # node_name: user-specified key for output dict
            'flatten': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_resnet152, return_nodes)
    elif network_name == "swin_v2_t":
        nn_model_swin_v2_t = models.swin_v2_t(weights="IMAGENET1K_V1")
        return_nodes = {
            'flatten': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_swin_v2_t, return_nodes)
    elif network_name == "swin_v2_s":
        nn_model_swin_v2_s = models.swin_v2_s(weights="IMAGENET1K_V1")
        return_nodes = {
            'flatten': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_swin_v2_s, return_nodes)
    elif network_name == "swin_v2_b":
        nn_model_swin_v2_b = models.swin_v2_b(weights="IMAGENET1K_V1")
        return_nodes = {
            'flatten': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_swin_v2_b, return_nodes)
    elif network_name == "vit_l_16":
        nn_model_vit_l_16 = models.vit_l_16(weights="IMAGENET1K_SWAG_LINEAR_V1")
        return_nodes = {
            'getitem_5': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_vit_l_16, return_nodes)
    elif network_name == "vit_h_14":
        nn_model_vit_h_14 = models.vit_h_14(weights="IMAGENET1K_SWAG_LINEAR_V1")
        return_nodes = {
            'getitem_5': 'feature',
        }
        nn_model = create_feature_extractor(nn_model_vit_h_14, return_nodes)
    else:
        raise NotImplementedError(network_name)
    nn_model = nn_model.to(host_device)
    nn_model.eval()

    if dataset_name == "stl10":
        transform_stl10 = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_set = datasets.STL10(root=dataset_path, split='train', transform=transform_stl10, download=True)
        test_set = datasets.STL10(root=dataset_path, split='test', transform=transform_stl10, download=True)
        val_number = int(len(train_set.data) * 0.4)
    elif dataset_name == "tiny-imagenet-200":
        transform_tinyimagenet = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_set = Tiny_ImageNet(root=dataset_path, split="train", transform=transform_tinyimagenet)
        test_set = Tiny_ImageNet(root=dataset_path, split="test", transform=transform_tinyimagenet)
        val_number = int(len(train_set.data) * 0.4)
    elif dataset_name == "cifar10":
        transform_cifar10 = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
        train_set = datasets.CIFAR10(root=dataset_path, train=True, transform=transform_cifar10, download=True)
        test_set = datasets.CIFAR10(root=dataset_path, train=False, transform=transform_cifar10, download=True)
        val_number = int(len(train_set.data) * 0.4)
    elif dataset_name == "cifar100":
        transform_cifar100 = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
        ])
        train_set = datasets.CIFAR100(root=dataset_path, train=True, transform=transform_cifar100, download=True)
        test_set = datasets.CIFAR100(root=dataset_path, train=False, transform=transform_cifar100, download=True)
        val_number = int(len(train_set.data) * 0.4)
    elif dataset_name == "place365":
        num_classes = 365
        transform_place365 = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        try:
            train_set = datasets.Places365(root=dataset_path, split='train-standard', small=True,
                                           transform=transform_place365, download=True)
        except:
            train_set = datasets.Places365(root=dataset_path, split='train-standard', small=True,
                                           transform=transform_place365)
        train_num_per_class = 1000
        val_number_per_class = 100
        train_val_number_per_class = train_num_per_class + val_number_per_class
        try:
            test_set = datasets.Places365(root=dataset_path, split='val', small=True, transform=transform_place365,
                                          download=True)
        except:
            test_set = datasets.Places365(root=dataset_path, split='val', small=True, transform=transform_place365)

        # train set
        print("Extracting train set features...")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        nn_feature_train_list = []
        target_train_list = []
        for idx, data in enumerate(train_loader):
            image, target = data
            image = image.to(host_device)
            nn_results = nn_model(image)
            nn_feature = nn_results['feature'].detach().cpu().numpy()

            nn_feature_train_list.append(nn_feature)
            target_train_list.extend(target.tolist())

            idx_thold = num_classes * train_val_number_per_class / batch_size
            if idx > idx_thold and (idx % 50 == 0):
                if (sorted(Counter(target_train_list).values())[0] >= train_val_number_per_class):
                    break
            if (idx % 50 == 0): print(idx)

        nn_feature_train_val_np = np.vstack(nn_feature_train_list)
        target_train_val_np = np.array(target_train_list)

        # separate train and val data
        nn_feature_real_train_list = []
        target_real_train_list = []
        nn_feature_real_val_list = []
        target_real_val_list = []
        for class_i in range(num_classes):
            where_class_i = np.where(target_train_val_np == class_i)[0]
            assert (where_class_i.shape[0] >= train_val_number_per_class)
            # real train
            nn_feature_real_train_list.append(nn_feature_train_val_np[where_class_i[:train_num_per_class]])
            local_target_real_train_np = target_train_val_np[where_class_i[:train_num_per_class]]
            assert ((local_target_real_train_np == class_i).all())
            target_real_train_list.extend(local_target_real_train_np.tolist())
            # real val
            nn_feature_real_val_list.append(
                nn_feature_train_val_np[where_class_i[train_num_per_class:train_val_number_per_class]])
            local_target_real_val_np = target_train_val_np[
                where_class_i[train_num_per_class:train_val_number_per_class]]
            assert ((local_target_real_val_np == class_i).all())
            target_real_val_list.extend(local_target_real_val_np.tolist())

        nn_feature_real_train_np = np.vstack(nn_feature_real_train_list)
        target_real_train_np = np.array(target_real_train_list)
        nn_feature_real_val_np = np.vstack(nn_feature_real_val_list)
        target_real_val_np = np.array(target_real_val_list)

        save_train_dict = {"nn_feature_np": nn_feature_real_train_np, "target_np": target_real_train_np}
        num_train = target_real_train_np.shape[0]
        save_val_dict = {"nn_feature_np": nn_feature_real_val_np, "target_np": target_real_val_np}
        num_val = target_real_val_np.shape[0]

        # test set
        print("Extracting test set features...")
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
        nn_feature_val_test_list = []
        target_val_test_list = []
        for idx, data in enumerate(test_loader):
            image, target = data
            image = image.to(host_device)
            nn_results = nn_model(image)
            nn_feature = nn_results['feature'].detach().cpu().numpy()

            nn_feature_val_test_list.append(nn_feature)
            target_val_test_list.extend(target.tolist())
            if (idx % 50 == 0): print(idx)

        nn_feature_val_test_np = np.vstack(nn_feature_val_test_list)
        target_val_test_np = np.array(target_val_test_list)
        save_test_dict = {"nn_feature_np": nn_feature_val_test_np, "target_np": target_val_test_np}
        num_test = target_val_test_np.shape[0]
    else:
        raise NotImplementedError(dataset_name)

    def _feature_extractor(key, data_loader):
        nn_feature_train_list = []
        target_train_list = []
        for _, data in enumerate(data_loader):
            image, target = data
            if key == "target":
                target_train_list.extend(target.tolist())
                continue
            image = image.to(host_device)
            nn_results = nn_model(image)
            nn_feature = nn_results[key].detach().cpu().numpy()
            nn_feature_train_list.append(nn_feature)

        if key == "target":
            return np.array(target_train_list)
        return np.vstack(nn_feature_train_list)

    # train set
    print("Extracting train set features...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    # random shuffle train set
    if os.path.exists(dataset_path + f"training_validation_indices_{val_number}.pkl"):
        print(f"Loading training_validation_indices_{val_number}.pkl...")
        train_idxs, val_idxs = pickle.load(open(dataset_path + f"training_validation_indices_{val_number}.pkl", "rb"))
    else:
        print(f"Generating training_validation_indices_{val_number}.pkl...")
        shuffled_val_train_idxs = np.random.permutation(len(train_loader.dataset))
        val_idxs = shuffled_val_train_idxs[:val_number]
        train_idxs = shuffled_val_train_idxs[val_number:]
        pickle.dump((train_idxs, val_idxs), open(dataset_path + f"training_validation_indices_{val_number}.pkl", "wb"))

    save_train_dict = {"nn_feature_np": _feature_extractor("feature", train_loader)[train_idxs],
                       "target_np": _feature_extractor("target", train_loader)[train_idxs]}
    num_train = save_train_dict["target_np"].shape[0]
    pickle.dump(save_train_dict,
                open(
                    dataset_path + "training_" + str(num_train) + "_Rseed_" + str(r_seed) + version_suffix + ".pkl",
                    "wb"))
    print("Save train set features DONE!")
    del save_train_dict
    gc.collect()

    # val set
    save_val_dict = {"nn_feature_np": _feature_extractor("feature", train_loader)[val_idxs],
                     "target_np": _feature_extractor("target", train_loader)[val_idxs]}
    num_val = save_val_dict["target_np"].shape[0]
    pickle.dump(save_val_dict,
                open(
                    dataset_path + "validation_" + str(num_val) + "_Rseed_" + str(r_seed) + version_suffix + ".pkl",
                    "wb"))
    print("Save val set features DONE!")
    del save_val_dict
    gc.collect()

    # test set
    print("Extracting test set features...")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    save_test_dict = {"nn_feature_np": _feature_extractor("feature", test_loader),
                      "target_np": _feature_extractor("target", test_loader)}
    num_test = save_test_dict["target_np"].shape[0]
    pickle.dump(save_test_dict,
                open(dataset_path + "testing_" + str(num_test) + "_Rseed_" + str(r_seed) + version_suffix + ".pkl",
                     "wb"))
    print("Save test set features DONE!")
    del save_test_dict
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, type=str, help="network name")
    parser.add_argument("--dataset", required=True, type=str, help="dataset name")
    in_args = parser.parse_args()

    network_name = in_args.network
    if network_name not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "swin_v2_t", "swin_v2_s",
                            "swin_v2_b", "vit_l_16", "vit_h_14"]:
        raise NotImplementedError(network_name)

    dataset_name = in_args.dataset
    if dataset_name not in ["stl10", "cifar10", "cifar100", "tiny-imagenet-200"]:
        raise NotImplementedError(dataset_name)

    args = {
        "dataset_name": dataset_name,
        "dataset_path": f"datasets/{dataset_name}/",
        "network_name": network_name,
        "network_path": f"networks/{network_name}/",
        "r_seed": 0,
        "batch_size": 1,
        "num_workers": 0,
        "version_suffix": f"_{network_name}",
    }
    main(args)
