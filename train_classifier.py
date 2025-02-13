import torch
from torch.utils.data import DataLoader
import os
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
    feature_type = args["feature_type"]
    classifier_path = args["classifier_path"]
    r_seed = args["r_seed"]
    batch_size = args["batch_size"]
    num_workers = args["num_workers"]
    learning_rate = args["learning_rate"]
    num_epochs = args["num_epochs"]
    val_every_epoch = args["val_every_epoch"]

    # set random seed
    torch.manual_seed(r_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(classifier_path): os.makedirs(classifier_path)
    host_device = "cuda" if torch.cuda.is_available() else "cpu"

    # get data sets and data loaders
    train_set, val_set, test_set = get_datasets(dataset_name, dataset_path, network_name, feature_type)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # get model & set optimizer and loss function
    nn_classifier = get_classifier(network_name, dataset_name, feature_type)
    nn_classifier = nn_classifier.to(host_device)
    optimizer = torch.optim.Adam(nn_classifier.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    tb_writer = SummaryWriter(log_dir=classifier_path)

    best_val_epoch_acc = -1
    for epoch in range(1, num_epochs + 1):
        # training
        nn_classifier.train()
        train_loss_list = []
        train_accuracy_list = []
        for _, data in enumerate(train_loader):
            x_feature = data["feature"].to(host_device)
            x_target = data["target"].to(host_device)

            optimizer.zero_grad()
            x_logit = nn_classifier(x_feature)
            train_loss = loss_func(x_logit, x_target)
            train_loss.backward()
            optimizer.step()

            # log training stats
            train_loss_list.append(train_loss.item())
            train_accuracy_list.append(((x_logit.argmax(dim=-1) == x_target).sum() / x_target.shape[0]).item())

        train_epoch_loss = np.mean(train_loss_list)
        train_epoch_accuracy = np.mean(train_accuracy_list)
        print("Epoch: %d, Train - Loss: %.3f, Acc: %.2f" % (epoch, train_epoch_loss, train_epoch_accuracy * 100))
        tb_writer.add_scalar('Train/loss', train_epoch_loss, epoch)
        tb_writer.add_scalar('Train/accuracy', train_epoch_accuracy, epoch)

        # validation
        if epoch % val_every_epoch == 0:
            nn_classifier.eval()
            val_loss_list = []
            val_accuracy_list = []
            x_feature_list = []
            x_target_list = []
            x_logit_list = []
            for _, data in enumerate(val_loader):
                x_feature = data["feature"].to(host_device)
                x_target = data["target"].to(host_device)
                x_logit = nn_classifier(x_feature)

                x_feature_list.append(x_feature)
                x_target_list.append(x_target)
                x_logit_list.append(x_logit)

                val_loss_list.append(loss_func(x_logit, x_target).item())
                val_accuracy_list.append(((x_logit.argmax(dim=-1) == x_target).sum() / x_target.shape[0]).item())

            val_epoch_loss = np.mean(val_loss_list)
            val_epoch_accuracy = np.mean(val_accuracy_list)
            print("----- Epoch: %d, Validation - Loss: %.3f, Acc: %.2f" % (
                epoch, val_epoch_loss, val_epoch_accuracy * 100))
            tb_writer.add_scalar('Val/loss', val_epoch_loss, epoch)
            tb_writer.add_scalar('Val/accuracy', val_epoch_accuracy, epoch)

            # save the best val model
            if val_epoch_accuracy > best_val_epoch_acc:
                best_val_epoch_acc = val_epoch_accuracy
                saving_stats = {
                    "best_val_epoch_acc": val_epoch_accuracy,
                    "best_val_epoch": epoch,
                    "correspond_val_epoch_loss": val_epoch_loss,
                    "correspond_train_epoch_loss": train_epoch_loss,
                    "correspond_train_epoch_acc": train_epoch_accuracy,
                }
                torch.save(nn_classifier.state_dict(), classifier_path + "/best_val_classifier.pt")
                pickle.dump(saving_stats, open(classifier_path + "/best_val_stats.pkl", "wb"))

                # torch.save(x_feature_list, classifier_path + "/val_feature_list.pt")
                torch.save(x_target_list, classifier_path + "/val_target_list.pt")
                torch.save(x_logit_list, classifier_path + "/val_logit_list.pt")

    # test
    nn_classifier.load_state_dict(torch.load(classifier_path + "/best_val_classifier.pt"))
    nn_classifier.eval()
    test_loss_list = []
    test_accuracy_list = []
    x_feature_list = []
    x_target_list = []
    x_logit_list = []
    for _, data in enumerate(test_loader):
        x_feature = data["feature"].to(host_device)
        x_target = data["target"].to(host_device)
        x_logit = nn_classifier(x_feature)

        x_feature_list.append(x_feature)
        x_target_list.append(x_target)
        x_logit_list.append(x_logit)

        test_loss_list.append(loss_func(x_logit, x_target).item())
        test_accuracy_list.append(((x_logit.argmax(dim=-1) == x_target).sum() / x_target.shape[0]).item())

    test_epoch_loss = np.mean(test_loss_list)
    test_epoch_accuracy = np.mean(test_accuracy_list)
    print("----- Test - Loss: %.3f, Acc: %.2f" % (test_epoch_loss, test_epoch_accuracy * 100))
    tb_writer.add_scalar('Test/loss', test_epoch_loss)
    tb_writer.add_scalar('Test/accuracy', test_epoch_accuracy)

    # torch.save(x_feature_list, classifier_path + "/test_feature_list.pt")
    torch.save(x_target_list, classifier_path + "/test_target_list.pt")
    torch.save(x_logit_list, classifier_path + "/test_logit_list.pt")

    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", required=True, type=str, help="feature type")
    parser.add_argument("--network", required=True, type=str, help="network name")
    parser.add_argument("--dataset", required=True, type=str, help="dataset name")
    in_args = parser.parse_args()

    feature_type = in_args.feature
    if feature_type not in ["nn_conv1_maxpool_np", "nn_layer1_relu_2_np", "nn_layer2_relu_2_np",
                            "nn_layer3_relu_2_np", "nn_feature_np"]:
        raise NotImplementedError(feature_type)

    network_name = in_args.network
    if network_name not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                            "swin_v2_t", "swin_v2_s", "swin_v2_b", "vit_l_16", "vit_h_14"]:
        raise NotImplementedError(network_name)

    dataset_name = in_args.dataset
    if dataset_name not in ["stl10", "cifar10", "cifar100", "tiny-imagenet-200"]:
        raise NotImplementedError(dataset_name)

    args = {
        "dataset_name": dataset_name,
        "dataset_path": os.path.join("datasets", dataset_name),
        "network_name": network_name,
        "feature_type": feature_type,
        "classifier_path": os.path.join("classifiers", dataset_name,
                                        network_name + '_' +
                                        feature_type + datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")),
        "r_seed": 0,
        "batch_size": 500,
        "num_workers": 4,
        "learning_rate": 0.00001,
        "num_epochs": 1000,
        "val_every_epoch": 1,
    }
    main(args)
