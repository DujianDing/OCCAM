from torch.utils.data import Dataset
import os
import pickle


class Feature_Data_pkl(Dataset):
    def __init__(self, data_file_path, feature_type):
        data_file = pickle.load(open(data_file_path, "rb"))
        self.data_nn_feature = data_file[feature_type] # [N, 2048]
        self.data_target = data_file['target_np'] # [N]
        # a sanity check on dimension
        assert (self.data_nn_feature.shape[0] == self.data_target.shape[0])

    def __len__(self):
        return self.data_nn_feature.shape[0]

    def __getitem__(self, idx):
        feature = self.data_nn_feature[idx]
        target = self.data_target[idx]
        return {"feature": feature, "target": target}


def get_datasets(dataset_name, dataset_path, model_name, feature_type):
    if dataset_name == "stl10":
        train_data_file_path = os.path.join(dataset_path, f"training_3000_Rseed_0_{model_name}.pkl")
        train_dataset = Feature_Data_pkl(train_data_file_path, feature_type)
        val_data_file_path = os.path.join(dataset_path, f"validation_2000_Rseed_0_{model_name}.pkl")
        val_dataset = Feature_Data_pkl(val_data_file_path, feature_type)
        test_data_file_path = os.path.join(dataset_path, f"testing_8000_Rseed_0_{model_name}.pkl")
        test_dataset = Feature_Data_pkl(test_data_file_path, feature_type)
    elif dataset_name == "cifar10":
        train_data_file_path = os.path.join(dataset_path, f"training_30000_Rseed_0_{model_name}.pkl")
        train_dataset = Feature_Data_pkl(train_data_file_path, feature_type)
        val_data_file_path = os.path.join(dataset_path, f"validation_20000_Rseed_0_{model_name}.pkl")
        val_dataset = Feature_Data_pkl(val_data_file_path, feature_type)
        test_data_file_path = os.path.join(dataset_path, f"testing_10000_Rseed_0_{model_name}.pkl")
        test_dataset = Feature_Data_pkl(test_data_file_path, feature_type)
    elif dataset_name == "cifar100":
        train_data_file_path = os.path.join(dataset_path, f"training_30000_Rseed_0_{model_name}.pkl")
        train_dataset = Feature_Data_pkl(train_data_file_path, feature_type)
        val_data_file_path = os.path.join(dataset_path, f"validation_20000_Rseed_0_{model_name}.pkl")
        val_dataset = Feature_Data_pkl(val_data_file_path, feature_type)
        test_data_file_path = os.path.join(dataset_path, f"testing_10000_Rseed_0_{model_name}.pkl")
        test_dataset = Feature_Data_pkl(test_data_file_path, feature_type)
    elif dataset_name == "tiny-imagenet-200":
        train_data_file_path = os.path.join(dataset_path, f"training_60000_Rseed_0_{model_name}.pkl")
        train_dataset = Feature_Data_pkl(train_data_file_path, feature_type)
        val_data_file_path = os.path.join(dataset_path, f"validation_40000_Rseed_0_{model_name}.pkl")
        val_dataset = Feature_Data_pkl(val_data_file_path, feature_type)
        test_data_file_path = os.path.join(dataset_path, f"testing_10000_Rseed_0_{model_name}.pkl")
        test_dataset = Feature_Data_pkl(test_data_file_path, feature_type)
    elif dataset_name == "place365":
        train_data_file_path = os.path.join(dataset_path, "training_365000_Rseed_0_Q6K.pkl")
        train_dataset = Feature_Data_pkl(train_data_file_path)
        val_data_file_path = os.path.join(dataset_path, "validation_36500_Rseed_0_Q6K.pkl")
        val_dataset = Feature_Data_pkl(val_data_file_path)
    else:
        raise NotImplementedError(dataset_name)

    return train_dataset, val_dataset, test_dataset