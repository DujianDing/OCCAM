import torch.nn as nn


class Classifier_Net(nn.Module):
    def __init__(self, feature_type, input_dim, output_dim):
        super().__init__()
        if feature_type == "nn_conv1_maxpool_np":
            self.classifier = nn.Sequential(
                                  nn.AdaptiveAvgPool2d(output_size=(4, 8)), # nn.AdaptiveAvgPool2d(output_size=(8, 4)),
                                  nn.Flatten(),
                                  nn.Linear(input_dim, output_dim),
                              )
        elif feature_type == "nn_layer1_relu_2_np":
            self.classifier = nn.Sequential(
                                  nn.AdaptiveAvgPool2d(output_size=(2, 4)), # nn.AdaptiveAvgPool2d(output_size=(4, 2)),
                                  nn.Flatten(),
                                  nn.Linear(input_dim, output_dim),
                              )
        elif feature_type == "nn_layer2_relu_2_np":
            self.classifier = nn.Sequential(
                                  nn.AdaptiveAvgPool2d(output_size=(2, 2)),
                                  nn.Flatten(),
                                  nn.Linear(input_dim, output_dim),
                              )
        elif feature_type == "nn_layer3_relu_2_np":
            self.classifier = nn.Sequential(
                                  nn.AdaptiveAvgPool2d(output_size=(1, 2)), # nn.AdaptiveAvgPool2d(output_size=(2, 1)),
                                  nn.Flatten(),
                                  nn.Linear(input_dim, output_dim),
                              )
        elif feature_type == "nn_feature_np":
            self.classifier = nn.Linear(input_dim, output_dim)
        else:
            raise NotImplementedError(feature_type)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x_logit = self.classifier(x)
        return x_logit


def get_classifier(network_name, dataset_name, feature_type):
    if network_name == "resnet18":
        input_dim = 512
    elif network_name == "resnet34":
        input_dim = 512
    elif network_name == "resnet50":
        input_dim = 2048
    elif network_name == "resnet101":
        input_dim = 2048
    elif network_name == "resnet152":
        input_dim = 2048
    elif network_name == "swin_v2_t":
        input_dim = 768
    elif network_name == "swin_v2_s":
        input_dim = 768
    elif network_name == "swin_v2_b":
        input_dim = 1024
    elif network_name == "vit_l_16":
        input_dim = 1024
    elif network_name == "vit_h_14":
        input_dim = 1280
    else:
        raise NotImplementedError(network_name)

    if dataset_name == "stl10":
        output_dim = 10
    elif dataset_name == "cifar10":
        output_dim = 10
    elif dataset_name == "cifar100":
        output_dim = 100
    elif dataset_name == "tiny-imagenet-200":
        output_dim = 200
    else:
        raise NotImplementedError(dataset_name)

    nn_classifier = Classifier_Net(feature_type, input_dim, output_dim)
    return nn_classifier