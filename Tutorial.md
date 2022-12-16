---
layout: page
title: Tutorial

---

<br>
<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/Logo_v2_White_transparent.png?raw=true" width="500" align="center" /></center>

<h2 style="color:white;">Content</h2>

<ul>
  <li><a href="#File structure">File structure</a></li>
  <li><a href="#Data Pre-processing">How to generate training dataset</a></li>
  <li><a href="#Implementation of Python code">How to perform training and inference</a></li>
  <li><a href="#Fiji plugin">How to use our Fiji plugin</a></li>
</ul>

Our source code can be downloaded from <a href='https://github.com/'>https://github.com/</a>. We use MATLAB R2021b to generate training datasets from raw images, and TensorFlow to perform training and inference.

<h2 style="color:white;" id="File structure">1. File structure</h2>

- <code style="background-color:#393939;">./dataset</code> is the default path for training data generation and test data
  - <code style="background-color:#393939;">./dataset/WF_2D</code> includes raw 2D WF data
  - <code style="background-color:#393939;">./dataset/LLS_2D</code> includes raw 3D LLS data
  - <code style="background-color:#393939;">./dataset/test_WF_2D</code>includes demo image of Lamp1 to test ZS-DeconvNet 2D
  - <code style="background-color:#393939;">./dataset/test_LLS_3D</code>includes demo stack of Lamp1 to test ZS-DeconvNet 3D
  - <code style="background-color:#393939;">./dataset/test_confocal_3D</code>includes demo stack of Microtubules to test ZS-DeconvNet 3D
  - <code style="background-color:#393939;">./dataset/PSF</code> includes the 2D simulated and experimantal PSF, and the 3D simulated PSF used for training
- <code style="background-color:#393939;">./augmented_datasets</code> is the default augmented training dataset path
- <code style="background-color:#393939;">./data_augment_recorrupt_matlab</code> includes the MATLAB code for generating training datasets
- <code style="background-color:#393939;">./train_inference_python</code> includes the Python code of training and inference, and the required dependencies.
  - <code style="background-color:#393939;">./train_inference_python/models</code> includes the optional models
  - <code style="background-color:#393939;">./train_inference_python/utils</code> is the tool package
  - <code style="background-color:#393939;">./train_inference_python/saved_models</code> includes pre-trained models for testing, and is the default path to save your trained models

<h2 style="color:white;" id="Data pre-processing">2. How to generate training dataset</h2>

If you would like to use our provided augmented training datasets in the folder <code style="background-color:#393939;">ZS-DeconvNet/augmented_datasets/</code>, you can skip this part.

<h3 style="color:white;">2.1 Data Augmentation and Re-corruption for 2D Data</h3>

We use MATLAB R2021b but previous versions might be compatible. After cloning our source code, you can:

+ Run <code style="background-color:#393939;">ZS-DeconvNet/data_augment_recorrupt_matlab/DataAugmFor2D.m</code> in MATLAB to generate 2D training data. Re-corruption is embedded in the process of augmentation. 

+ The augmented training datasets will be saved to the folder <code style="background-color:#393939;">ZS-DeconvNet/your_augmented_datasets/WF_2D</code>.

+ The default option is to use the  provided datasets in the folder <code style="background-color:#393939;">ZS-DeconvNet/datasets/WF_2D</code>to generate training data. But you can use your own data or download more data from our [shared datasets](...). If you want to use your own data or download other data, organize the data in the same way we do, or change the data loading part in the MATLAB code.

<h3 style="color:white;">2.2 Data Augmentation and Re-sampling for 3D Data</h3>

Similar to training datasets generation with 2D data, you can:

+ Run <code style="background-color:#393939;">ZS-DeconvNet/data_augment_recorrupt_matlab/DataAugmFor3D.m</code> to generate 3D training data. Re-sampling is embedded in the process of augmentation. 

+ The augmented training datasets will be saved automatically to the folder <code style="background-color:#393939;">ZS-DeconvNet/your_augmented_datasets/LLS_3D</code>.

+ The default option is to use the provided datasets in the folder <code style="background-color:#393939;">ZS-DeconvNet/datasets/LLS_3D</code> to perform augmentation and re-sampling, but you can use other data. 

<h2 style="color:white;" id="Implementation of Python code">3. How to perform training and inference</h2>

<h3 style="color:white;">3.1 Building environment</h3>

Our environment is:

- Windows 10
- Python 3.9.7
- Tensorflow-gpu 2.5.0
- NVIDIA GPU (GeForce RTX 3090) + CUDA (11.4)

To use our code, you should create a virtual environment and install the required packages first.

<code style="background-color:#393939;">
$ conda create -n zs-deconvnet python=3.9.7 
</code>

<code style="background-color:#393939;">
$ conda activate zs-deconvnet
</code>

<code style="background-color:#393939;">
$ pip install -r requirements.txt
</code>

After that, remember to install the right version of CUDA and cuDNN, if you want to use GPU. You can get the compatible version by searching

<code style="background-color:#393939;">
$ conda search cudatoolkit --info
</code>

<code style="background-color:#393939;">
$ conda search cudnn --info
</code>

then installing

<code style="background-color:#393939;">
$ conda install cudatoolkit==xx.x.x
</code>

<code style="background-color:#393939;">
$ conda install ducnn==x.x.x
</code>

<h3 style="color:white;">3.2 Training Demo</h3>

If you have generated your own data following the instructions in the previous part:

+ Change the data paths in <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/train_demo_2D.sh</code>or <code style="background-color:#393939;">train_inference_python/train_demo_3D.sh</code>. 

+ Run it in your terminal.

+ The result wills be saved to <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/saved_models/</code>.

+ Run <code style="background-color:#393939;">tensorboard --logdir [save_weights_dir]/[save_weights_name]/graph</code> to monitor the training process via tensorboard if you want.

If you would rather just try out the training code and not generate any data, you could run <code style="background-color:#393939;">train_demo_2D.sh</code> or <code style="background-color:#393939;">train_demo_3D.sh</code>directly, for the default data paths point to the augmented training datasets we have prepared for you.

+ Notice: the padded size of training data should be the multiple of $2^{\text{conv_block_num}}$, to be compatible with the 2D U-net structure. Be careful if you are changing the parameters <code style="background-color:#393939;">input_x</code>, <code style="background-color:#393939;">input_y</code> or <code style="background-color:#393939;">insert_xy</code>.

<h3 style="color:white;">3.3 Inference Demo</h3>

If you have trained a network yourself and want to test it:

+ Change the model weight paths in <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/infer_demo_2D.sh</code> or <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/infer_demo_3D.sh</code> accordingly. 

+ Run it in your terminal.

+ The output will be automatically saved to the folder where you load weights. For example, if you load weights from <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/saved_models/.../weights_40000.h5</code>, then the output will be saved to <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/saved_models/.../Inference/</code>.

+ The default option is to use demo test datasets in the folder <code style="background-color:#393939;">ZS-DeconvNet/datasets/test_WF_2D</code>, <code style="background-color:#393939;">ZS-DeconvNet/datasets/test_confocal_3D</code> and <code style="background-color:#393939;">ZS-DeconvNet/datasets/test_LLS_3D</code>, but you can use other data.

Otherwise:

+ We have provided saved models in the folder<code style="background-color:#393939;"> ZS-DeconvNet/train_inference_python/saved_models/</code>, and they are the default loading weights paths.

+ Run <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/infer_demo_2D.sh</code> or <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/infer_demo_3D.sh</code> in your terminal.

+ The 2D WF output will be automatically saved to the folder <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/saved_models/WF_2D_560_beta1_0.5-1.5_beta2_10-15_alpha1-2_SegNum20000_twostage_Unet_Hess0.02/Inference</code>, 3D confocal output to <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/saved_models/Confocal_3D_488_twostage_RCAN3D_upsample/Inference/</code>, and 3D LLS output to <code style="background-color:#393939;">ZS-DeconvNet/train_inference_python/saved_models/LLS_3D_488_Zsize5_Xsize48_fromMRC_twostage_RCAN3D_Hess0.1_MAE_up/Inference</code>.

+ Notice: If you are using image segmentation and fusion, which may be needed when the test image is too large and the memory runs out, please make sure <code style="background-color:#393939;">input_x-overlap_x</code> is the multiple of <code style="background-color:#393939;">seg_window_x-overlap_x</code>, or the image fusion will go wrong. The same caution is needed when dealing with y or z directions.

<h2 style="color:white;" id="Fiji plugin">4. How to use our Fiji plugin</h2>

<h3 style="color:white;">4.1 Installation</h3>
Our Fiji release is included in the open-source code, you can follow the instructions below to install the plugin:

+ Copy <code style="background-color:#393939;">ZS-DeconvNet/Fiji-plugin/jars/*</code> and <code style="background-color:#393939;">ZS-DeconvNet/Fiji-plugin/plugins/*</code> to your root path of Fiji <code style="background-color:#393939;">/*/Fiji.app/</code>.

+ Restart Fiji.

<h3 style="color:white;">4.2 About GPU and TensorFlow version</h3>
The ZS-DeconvNet Fiji plugin was developed based on TensorFlow-Java 1.15.0, which is compatible with CUDA version of 10.1 and cuDNN version of 7.5.1. If you would like to process models with a different TensorFlow version, or running with different GPU settings, please do the following:

+ Open <i>Edit > Options > Tensorflow</i>, and choose the version matching your model or setting.

+ Wait until a message pops up telling you that the library was installed.

+ Restart Fiji.

<h3 style="color:white;">4.3 Inference with ZS-DeconvNet Fiji plugin</h3>

Given a pre-trained ZS-DeconvNet model and an image or stack to be processed, the Fiji plugin is able to generate the corresponding denoised (optional) and super-resolved deconvolution image or stack. The workflow includes following steps: 

+ Open the image or stack in Fiji and start ZS-DeconvNet plugin by Clicking <i>Plugins > ZS-DeconvNet > predict ZS-DeconvNet 2D / predict ZS-DeconvNet 3D</i>.

+ Select the network model file, i.e., .zip file in the format of BioImage Model Zoo bundle. Of note, the model file could be trained and saved either by Python codes (see [this gist](https://gist.github.com/asimshankar/000b8d276f211f972168afa138eb3cc7)) or ZS-DeconvNet Fiji plugin, but has to be saved with TensorFlow environment <= 1.15.0.

+ Check inference options and choose hyper-parameters used in the inference. The options and parameters here are primarily selected to properly normalize the input data (NormalizeInput, PercentileBottom, and PerventileTop), perform tiling prediction to save memory of CPUs or GPUs (Number of tiles, Overlap between tiles, and Batch size), and decide whether to show process dialog and denoising results or not (Show process dialog and Show denoising result). A detailed description table is shown below:
  
  | Hyper-parameter                    | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:----------------------------------:|:-------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
  | NormalizeInput                     | Yes           | If you tick this hyper-parameter, the image or stack to be processed will be normalized.                                                                                                                                                                                                                                                                                                                                                                                                    |
  | PercentileBottom, PercentileTop    | 0,100         | These two hyper-parameters are valid only when you tick NormalizeInput, under which circumstance the image or stack $x$ will be normalized to $\tilde{x}$ such that$\tilde{x} =  \frac{x-prctile(x,PercentileBottom)}{prctile(x,PercentileTop)-prctile(PercentileBottom)}, $where $prctile(x, Percentile)$ is a function that returns the Percentile% largest value in $x$. Then the pixels with value lower than 0 will be set to 0, and pixels with value larger than 1 will be set to 1. |
  | Number of tiles                    | 8             | The number of image(stack) sections that the image(stack) will be divided to. Each tile will be processed separately. When the image processing is done, all processed tiles will be fused together. This separate – fuse procedure is designed for the circumstances when the image or stack is too large, and is not necessary if your memory is enough. In the latter case, just set Number of tiles to 1, and the image or stack will not be segmented.                                 |
  | Overlap between tiles              | 32            | The overlapping size between adjacent tiles in pixels. Has to be big enough so that the edge of tiles can merge well.                                                                                                                                                                                                                                                                                                                                                                       |
  | Batch size                         | 1             | The batch size of inference network. Has to be smaller than the number of tiles. Bigger batch size takes up more memory, but accelerates the image processing.                                                                                                                                                                                                                                                                                                                              |
  | Import model (.zip)                |               | Click <i>Browse</i> to select the pre-trained model, or enter the root path in the box. <b>The pre-trained model has to be saved in Tensorflow <= 1.15.0 environment.</b>                                                                                                                                                                                                                                                                                                                   |
  | Adjust mapping of TF network input |               | Click if you want to adjust the input 3D stack from x-y-t to x-y-z. It is recommended that you click this button every time you want to process a 3D stack, unless you are very sure the stack is in x-y-z order.                                                                                                                                                                                                                                                                           |
  | Show progress dialog               | Yes           | Tick if you want to see the progress bar and the time elapse of image processing.                                                                                                                                                                                                                                                                                                                                                                                                           |
  | Show denoise result                | No            | Tick if you want to see the denoised output.                                                                                                                                                                                                                                                                                                                                                                                                                                                |

+ After image processing with status bar shown in the message box (if select Show process dialog), the denoised (if select Show denoising result) and deconvolved output will pop out in separate Fiji windows automatically. Then the processed images or stacks could be viewed, manipulated, and saved via Fiji.

<h3 style="color:white;">4.4 Training with ZS-DeconvNet Fiji plugin</h3>

For ZS-DeconvNet model training, we generally provide two commands: <i>Train on augmented data</i> and <i>Train on opened images</i>, which differ in the ways of data loading and augmentation. The former command loads input data and corresponding GT images which are augmented elsewhere, e.g., in MATLAB or Python, from two data folders file by file, and the latter command directly takes the image stack opened in the current Fiji window as the training data and automatically perform data augmentation including image re-corruption (for 2D cases), random cropping, rotation and flipping into a pre-specified patch number. 

The overall workflow of ZS-DeconvNet training with Fiji plugin includes following steps:

+ Open the image or stack to be used for training in Fiji and start the ZS-DeconvNet plugin by clicking <i>Plugins > ZS-DeconvNet > train on opened images</i>; or directly start the plugin by the alternative command <i>Plugins > ZS-DeconvNet > Train on augmented data</i> and select the folders containing input images, GT images, and validation images.

+ Select the network type, i.e., 2D ZS-DeconvNet or 3D ZS-DeconvNet, the PSF file used for calculating deconvolution loss and choose training hyper-parameters, which include total epochs, iteration number per epoch, batch size, and initial learning rate. For 2D ZS-DeconvNet training by the command of <i>train on opened images</i>, three extra recorruption-related parameters of $\alpha $, $\beta _1$, and $\beta _2$ are tuneable, where $\alpha $ and $\beta _1$ are set as [1, 2] and [0.5, 1.5] by default, and $\beta _2$ should be set as the standard deviation of the camera background, which could be pre-calibrated from blank frames or calculated from empty regions of the training data. A detailed description table of these hyper-parameters is shown below:

| Hyper-parameter                                             | Default value       | Description                                                                                              |
|:-----------------------------------------------------------:|:-------------------:|:--------------------------------------------------------------------------------------------------------:|
| Input image folder for training (if select train on folder) |                     | Root path of the input image or stack folder.                                                            |
| GT image folder for training (if select train on folder)    |                     | Root path of the GT image or stack folder.                                                               |
| Background of images                                        | ?                   | Pixel value of the mean background noise.                                                                |
| Alpha, beta?                                                |                     |                                                                                                          |
| PSF file                                                    |                     | Root path of the PSF file used for calculating deconvolution loss. The PSF size has to be an odd number. |
| Model to train                                              | 2D ZS-DeconvNet     | The network type for training.                                                                           |
| Weights of Hessian Reg.                                     | 1?                  | The weight of Hessian regularization term.                                                               |
| Total epochs                                                | 200                 | The number of training epochs.                                                                           |
| iteration number per epoch                                  | 200                 | The number of training iterations per epoch.                                                             |
| Batch size                                                  | 4                   | Batch size in training.                                                                                  |
| Patch shape                                                 | 128                 | The shape of the training data. Select from given number.                                                |
| Initial learning rate                                       | $0.5\times 10^{-4}$ | The initial learning rate.                                                                               |

+ Click OK to start training. During the training procedure, the training progress and current learning rate will be displayed in a message box, and the model will be validated after each training epoch with the validation input and output shown in another image window for reference. 

+ Three types of exit:
  
  (i) If you don't want to train or save this model anymore for certain reasons like you got the hyper-parameters wrong, press <i>Cancel > Close</i> to enforce an exit.
  
  (ii) If you want an early stop, press <i>Finish</i> to finish training progress and save the model by <i>File actions > Save to..</i>.
  
  (iii) If you have finished training, in <i>Overview > Metadata > inputs & outputs > Training</i>, you will see the parameters of the trained model. Press <i>Export Model</i> and save the model by <i>File actions > Save to..</i>.
  
  Of note, you can also press <i>Export Model</i> during training to export the lastest saved model without disposing the training progress.
