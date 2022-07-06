from phiseg.model_zoo import likelihoods, posteriors, priors
import tensorflow as tf
from tfwrapper import normalisation as tfnorm

experiment_name = 'phiseg_7_5_BRATS_Aug2'
log_dir_name = 'BRATS'

# architecture
posterior = posteriors.phiseg
likelihood = likelihoods.phiseg
prior = priors.phiseg
layer_norm = tfnorm.batch_norm
use_logistic_transform = False

latent_levels = 5
resolution_levels = 7
n0 = 16
zdim0 = 2
max_channel_power = 4  # max number of channels will be n0*2**max_channel_power

# Data settings
data_identifier = 'BRATS'
preproc_folder = "<Path-to-your-data>"
data_root = "<Path-to-your-data>"
dimensionality_mode = '2D'
image_size = (256, 256, 4)
nlabels = 4
num_labels_per_subject = 1

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': nlabels
    }

# training
optimizer = tf.train.AdamOptimizer
lr_schedule_dict = {0: 3e-4}#{0: 1e-3}
deep_supervision = True
batch_size = 11
num_iter = 2818 * 100
annotator_range = range(0,nlabels*num_labels_per_subject,nlabels)  # which annotators to actually use for training

# losses
KL_divergence_loss_weight = 1
exponential_weighting = True

residual_multinoulli_loss_weight = 1.0

# monitoring
do_image_summaries = True
rescale_RGB = False
validation_frequency = 2818
validation_samples = 3
validation_batchsize = 5
num_validation_images = 'all' #'all'
tensorboard_update_frequency = 100