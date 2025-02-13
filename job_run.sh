#!/bin/bash
#SBATCH --time=0-8:00
#SBATCH --account=Your_Account
#SBATCH --job-name=Your_Job_Name
#SBATCH --output=Your_Job_Name.out
#SBATCH --mail-user=Your_Email
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mem=60G
#SBATCH --gpus-per-node=v100l:1


lscpu
hostname
nvidia-smi

module load StdEnv/2020 gcc/9.3.0 python/3.8.10 cuda/11.2.2

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install torch==1.13.0 torchvision==0.14.0 --no-index
pip install numpy==1.22.2 --no-index
pip install tensorboard==2.13.0 --no-index
pip install six==1.16.0 --no-index
pip install scipy --no-index

# network_name in ["resnet18", "resnet34", "resnet50", "resnet101", "swin_v2_t", "swin_v2_s", "swin_v2_b"]
# dataset_name in ["cifar10", "cifar100", "tiny-imagenet-200"]
python extract_feature.py --network "swin_v2_t" --dataset "tiny-imagenet-200"


# network_name in ["resnet18", "resnet34", "resnet50", "resnet101", "swin_v2_t", "swin_v2_s", "swin_v2_b"]
# dataset_name in ["cifar10", "cifar100", "tiny-imagenet-200"]
python train_classifier.py --feature "nn_feature_np" --network "swin_v2_t" --dataset "tiny-imagenet-200"


# imagenet inference
# network_name in ["resnet18", "resnet34", "resnet50", "resnet101", "swin_v2_t", "swin_v2_s", "swin_v2_b"]
python imagenet_classifier_inference.py --network "swin_v2_t" --dataset "imagenet"

