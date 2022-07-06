import os

########Dataset Configs############
dataset_name = "BRATS"
subset = 0.6
dataset_path = "<Path-to-your-data>"
num_workers = 11
ignore_index = 255
num_classes = 4
num_channels = 4

########Model Configs##############
base_model_name = "Unet"
model_type = "Dec2"   # VIMH
# model_type = "Dec2" # MH
prior = 0.1
klfactor = 1e-8
NUM_MODELS = 12 # VIMH=4, MH=12
LAMBDA = 0.5

########Train Configs##############
resume_path = None  # 'saves/{str_name}/best_train.save'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

best_train = 0.25

start_epoch = 0
epochs = 200

num_train_samples = 3
batch_size = 11

lr = 3e-4  # Karpathy constant

########Val Configs################
best_eval = 0.25
num_eval_samples = 9
num_plot_samples = 101
val_batch_size = 11

########Logging Configs############
cm_cuda = True
str_name = f"{dataset_name}_{base_model_name}_{model_type}_NM{NUM_MODELS}_L{LAMBDA}_kl{klfactor}_BS{batch_size}_lr{lr}"
