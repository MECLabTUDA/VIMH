# Practical uncertainty quantification of brain tumor segmentation 
Accepted at MIDL 2022, please see the paper & reviews:
https://openreview.net/forum?id=Srl3-HnY14U

if you use  parts of this work in your work please reference

>@inproceedings{
>fuchs2022practical, <br />
>title={Practical uncertainty quantification for brain tumor segmentation},<br />
>author={Moritz Fuchs and Camila Gonzalez and Anirban Mukhopadhyay},<br />
>booktitle={Medical Imaging with Deep Learning},<br />
>year={2022},<br />
>url={https://openreview.net/forum?id=Srl3-HnY14U } <br />
>}

## Abstract
Despite U-Nets being the de-facto standard for medical image segmentation, researchers have identified shortcomings of U-Nets, such as overconfidence and poor out-of-distribution generalization. Several methods for uncertainty quantification try to solve such problems by relying on well-known approximations such as Monte-Carlo Drop-Out, Probabilistic U-Net, and Stochastic Segmentation Networks. We introduce a novel multi-headed Variational U-Net. The proposed approach combines the global exploration capabilities of deep ensembles with the out-of-distribution robustness of Variational Inference. An efficient training strategy and an expressive yet general design ensure superior uncertainty quantification within a reasonable compute requirement. We thoroughly analyze the performance and properties of our approach on the publicly available BRATS2018 dataset. Further, we test our model on four commonly observed distribution shifts.
The proposed approach has good uncertainty calibration and is robust to out-of-distribution shifts.
## Installation
This framework was with the limitations of a *NVIDIA GeForce GTX 1080 TI* in mind so please adjust the Installation to your setup.

- Option 1: Installation with **requirements.txt** may be used to create an environment using:
    > $ conda create --name VIMH --file requirements.txt

- Option 2: manually install environment to you system e.g.:
    > $ conda create -n VIMH python=3.8.0<br />
    $ conda activate VIMH <br />
    $ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge<br />
    $ conda install -c simpleitk simpleitk<br />
    $ conda install -c conda-forge matplotlib tensorboardX tqdm
## Usage

To run the train scripts for a model run the *train_BRATS_Ensemble.py* with the appropiate config file. e.g.

> (VIMH) $ python train_BRATS_Ensemble.py ./config/VIMH.py

This framework currently supports following models:
>- VIMH 0.0
>- VIMH 0.5
>- VIMH 1.0
>- MH 0.0
>- MH 0.5
>- MH 1.0
>- [SSNs](https://github.com/biomedia-mira/stochastic_segmentation_networks)

and provides configuration files for other frameworks models:

> [Prob. U-net](https://github.com/SimonKohl/probabilistic_unet) <br />
> [PHiSeg](https://github.com/baumgach/PHiSeg-code)
## Acknowledgements

This work was supported by the Bundesministerium f ̈ur Gesundheit (BMG) with grant
[ZMVI1-2520DAT03A].