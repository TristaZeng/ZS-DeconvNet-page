---
layout: page
title: Tutorial

---

<br>
<center><img src="https://github.com/TristaZeng/ZS-DeconvNet-page/blob/page/images/Logo_v2_White_transparent.png?raw=true" width="500" align="center" /></center>

<p>Our source code can be downloaded from <a href='https://github.com/TristaZeng/ZS-DeconvNet'>https://github.com/TristaZeng/ZS-DeconvNet</a>. We use MATLAB R2021b to generate training datasets from raw images, and TensorFlow to perform training and inference. We also develop a Fiji plugin for both data generation and model training and inference. The detailed features are summarized in the table below:</p>

| Environment                    | Data augmentation | Network training|
  |:----------------------------------:|:-------------:|:-------------:|
  | MATLAB                    | 1. Save augmented images in local folders<br>2. Convenient for visualizing and checking the augmented data| / |
  | Python                    | / | 1. Faster than Fiji by ~1.6-fold<br>2. Automatically save trained models |
  | Fiji plugin                    | 1. Simple operation<br>2. Save augmented images in memory and directly train new models with them| 1. Simple operation<br>2. Automatically show the intermediate results<br>3. Save the models manually during the training process|



<h2 style="color:white;">Content</h2>

<ul>
  <li><a href="#File structure">File structure</a></li>
  <li><a href="#params">Detailed description of parameters</a></li>
  <li><a href="#Data Pre-processing">How to generate training dataset</a></li>
  <li><a href="#Implementation of Python code">How to perform training and inference</a></li>
  <li><a href="#Fiji plugin">How to use our Fiji plugin</a></li>
</ul>

<hr>

<h2 style="color:white;" id="File structure">1. File structure</h2>

In the folder <code style="background-color:#393939;">Python_MATLAB_Codes</code>,

- <code style="background-color:#393939;">./data_augment_recorrupt_matlab</code> includes the MATLAB codes for generating training datasets
- <code style="background-color:#393939;">./train_inference_python</code> includes the Python codes of training and inference, and the required dependencies
  - <code style="background-color:#393939;">./train_inference_python/models</code> includes the optional models
  - <code style="background-color:#393939;">./train_inference_python/utils</code> is the tool package

It is recommended to download the demo test data and pre-trained models <code style="background-color:#393939;">saved_models</code> from [our open-source datasets](replace_with_zenodo_path), and place it under the same folder so that:

- <code style="background-color:#393939;">./saved_models</code> includes pre-trained models for testing, and for each modality and structure <code style="background-color:#393939;">xx</code>:
  
  - <code style="background-color:#393939;">./saved_models/xx/saved_model</code> includes corresponding pre-trained model and inference result
  - <code style="background-color:#393939;">./saved_models/xx/test_data</code> includes raw test data

In the folder <code style="background-color:#393939;">Fiji_Plugin</code>,



<hr>

<h2 style="color:white;" id="params">2. Detailed description of parameters</h2>

| Hyper-parameter                                             | Suggested value       | Description                                                                                              |
|:-----------------------------------------------------------:|:-------------------:|:--------------------------------------------------------------------------------------------------------:|
| Background of images        | 100                  | Pixel value of the mean background noise.                                                                |
| Alpha for recorruption   | 1-2 | The noise magnification factor, which controls the overall magnitude of the added noises. The value of α does not affect the independence of the noise in the paired recorrupted images, thereby any values are theoretically applicable. However, in practice, to avoid over-corruption for either the input or target images, we adopted a modest range of [1, 2], for all 2D ZS-DeconvNet models, which is applicable for both simulated and experimental dataset of various specimens and imaging conditions in this paper.              |
| Beta1 for recorruption  | 0.5-1.5 | The Poissonian factor that affects the variance of the signal-dependent shot noise in the image recorruption process for 2D ZS-DeconvNet. The theoretically optimal value is 1. Nevertheless, we found that for experimental data, a random value within a small range, e.g., [0.5, 1.5], for each training patch pairs in the recorruption process achieves a stronger robustness and is applicable for various biological specimens and imaging conditions.              |
| Beta2 for recorruption   | estimated from data | The Gaussian factor that represents the variance of the additive Gaussian noises, i.e., the readout noise of the camera, which can be estimated from the sample-free region of the images in training dataset or pre-calibrated from the camera following standard protocols.               |
| PSF file                                                    |    /                 | Root path of the point spread function file used for calculating deconvolution loss. The PSF size has to be an odd number. The best option of the PSF is the measured beads because the experimentally acquired beads describe the actual imaging process best. But if the imaging system is well calibrated, i.e., with the least optical aberrations and its PSF is very close to theoretical one, the simulated PSF (e.g., generated via PSF Generator plugin) can be applied as well. We included the corresponding PSF for each type of data in our [open-source datasets](replace_with_zenodo_path). Of note, the PSF is normalized before the calculation by dividing the summation of its intensity in the software to ensure the output deconvolved image is conservative in terms of intensity.|
| Total number of augmentation                                                    |    20,000 (2D) / 10,000 (3D)                | The desired number of training patches after augmentation. |
| Model to train                                              | 2D ZS-DeconvNet/3D ZS-DeconvNet     | The network type for training.                                                                           |
| Weights of Hessian Reg.                                     | 0.02 (2D) / 0.1 (3D)                 | The scalar weight to balance the Hessian regularization in the loss function. The Hessian regularization used in the training process of ZS-DeconvNet is mainly to mitigate the slight pixelized artifact, therefore does not need to be tuned.                                                               |
| Total epochs                                                | 250 (2D) / 100 (3D)                | The number of training epochs.                                                                           |
| iteration number per epoch                                  | 200 (2D) / 100 (3D)               | The number of training iterations per epoch.                                                             |
| Batch size                                                  | 4 (2D) / 3 (3D)                  | The batch size is defined as the number of samples used for each training iteration, which mainly affects the convergence speed and generalization of the network models. Generally, a batch size that is either too large or too small may raise difficulties in the training procedure, e.g., out of memory error or unstable convergence. In our experiments of this paper, we adopted a modest batch size of 4 for 2D models and 3 for 3D models to balance the convergence speed, memory usage, and the model generalization, which are applied in all our experiments of this paper and robust enough for most applications.                                                                                 |
| Patch shape                                                 | 128x128 (2D) / 64x64x13 (3D)                 | The patch size determines the image shape after data augmentation, which may affect the total training time and final performance of the trained network models. We have tested the training time and performance of 2D/3D ZS-DeconvNet models trained with dataset of different patch sizes. We typically choose a relatively small patch size of 128×128 pixels for 2D models and 64×64×13 voxels for 3D models to speed up the training process, because the training duration goes longer as the training patch size gets larger but without obvious increasement in validation PSNR.                                               |
| Initial learning rate                                       | $0.5\times 10^{-4}$ (2D) / $1\times 10^{-4}$ (3D) | The learning rate determines the step size at each iteration during the network training, which decays by a factor of 0.5 every 50 epochs from the initial value in the implementations of ZS-DeconvNet. A higher initial learning rate typically leads to faster convergence of the model, while destabilizes the training process. Therefore, we empirically set the initial learning rate as $0.5\times 10^{-4}$ and $1\times 10^{-4}$ for 2D and 3D models, respectively.  |

<hr>

<h2 style="color:white;" id="Data pre-processing">3. How to generate training dataset</h2>

We use MATLAB R2021b but previous versions might be compatible. After cloning our source code, you can:

+ <p>Prepare a folder of raw data. Download <a href='replace_with_zenodo_path'>our open-source raw data</a> of various modalities or use your own raw data.</p> 

+ <p>Open <code style="background-color:#393939;">./data_augment_recorrupt_matlab/demo_augm.m</code> and replace the parameter <code style="background-color:#393939;">data_folder</code> with your raw data directory.</p>

+ <p>The default output path is <code style="background-color:#393939;">./your_augmented_datasets/</code>.</p>

<hr>

<h2 style="color:white;" id="Implementation of Python code">4. How to perform training and inference</h2>

<h3 style="color:white;">4.1 Building environment</h3>

Our environment is:

- Windows 10
- Python 3.9.7
- Tensorflow-gpu 2.5.0
- NVIDIA GPU (GeForce RTX 3090) + CUDA (11.4)

To use our code, you should create a virtual environment and install the required packages first.

<code style="background-color:#393939;">
$ conda create -n zs-deconvnet python=3.9.7 
</code><br>
<code style="background-color:#393939;">
$ conda activate zs-deconvnet
</code><br>
<code style="background-color:#393939;">
$ pip install -r requirements.txt
</code>

<p>After that, remember to install the right version of CUDA and cuDNN, if you want to use GPU. You can get the compatible version by searching</p>

<code style="background-color:#393939;">
$ conda search cudatoolkit --info
</code><br>
<code style="background-color:#393939;">
$ conda search cudnn --info
</code>

then installing

<code style="background-color:#393939;">
$ conda install cudatoolkit==xx.x.x
</code><br>
<code style="background-color:#393939;">
$ conda install cudnn==x.x.x
</code>

<h3 style="color:white;">4.2 Training Demo</h3>

Skip this part if you do not wish to train a new model. You can just test the demo test data using our provided pre-trained models. 

To train a new model, you need to:

+ <p>Generated the training dataset following the instructions in the previous part.</p>
+ <p>Choose a test image/volume and obtain the path to the corresponding PSF.</p>
+ <p>Change <code style="background-color:#393939;">otf_or_psf_path</code> (or <code style="background-color:#393939;">psf_path</code> in the case of 3D), <code style="background-color:#393939;">data_dir</code>, <code style="background-color:#393939;">folder</code> and <code style="background-color:#393939;">test_images_path</code> in <code style="background-color:#393939;">./train_inference_python/train_demo_2D.sh</code> or <code style="background-color:#393939;">train_inference_python/train_demo_3D.sh</code>.</p>
+ <p>Run it in your terminal.</p>
+ <p>The result wills be saved to <code style="background-color:#393939;">./your_saved_models/</code>.</p>
+ <p>Run <code style="background-color:#393939;">tensorboard --logdir [save_weights_dir]/[save_weights_name]/graph</code> to monitor the training process via tensorboard if needed.</p>
+ <p>Other <b>detailed description of each input argument of the python codes</b> can be found in the comments of <code style="background-color:#393939;">./train_inference_python/train_demo_2D.sh</code> or <code style="background-color:#393939;">train_inference_python/train_demo_3D.sh</code>.</p>

<h3 style="color:white;">4.3 Inference Demo</h3>

To test a well-trained ZS-DeconvNet model, you should:

+ Change the weight paths in <code style="background-color:#393939;">./train_inference_python/infer_demo_2D.sh</code> or <code style="background-color:#393939;">./train_inference_python/infer_demo_3D.sh</code> accordingly, or just use the default options given by us. 
+ Run it in your terminal.
+ The output will be saved to the folder where you load weights, e.g., if you load weights from <code style="background-color:#393939;">./train_inference_python/saved_models/.../weights_40000.h5</code>, then the output will be saved to <code style="background-color:#393939;">./train_inference_python/saved_models/.../Inference/</code>.

<hr>

<h2 style="color:white;" id="Fiji plugin">5. How to use our Fiji plugin</h2>

We provide a simple instruction for our plugin here. For latest updates, detailed parameter table and snapshots of usage, see the ReadMe.md in https://github.com/TristaZeng/ZS-DeconvNet/tree/main/Fiji_Plugin.

<h3 style="color:white;">5.1 Installation</h3>
Our Fiji release is included in the open-source code, you can follow the instructions below to install the plugin:

+ Copy <code style="background-color:#393939;">./Fiji-plugin/jars/*</code> and <code style="background-color:#393939;">./Fiji-plugin/plugins/*</code> to your root path of Fiji <code style="background-color:#393939;">/*/Fiji.app/</code>.
+ Restart Fiji.

We mainly developed and tested the ZS-DeconvNet Fiji plugin on workstations of Linux and Windows operating system equipped with Nvidia graphics cards. Because TensorFlow-GPU package is currently incompatible with MacOS, we are sorry that MacBook users can only use the TensorFlow-CPU to run our ZS-DeconvNet Fiji plugin at present, which is relatively inefficient compared to Nvidia GPU-based computation. We’ll be looking for the solutions and trying to make our plugin compatible with MacBook for higher efficiency in the future.

<h3 style="color:white;">5.2 About GPU and TensorFlow version</h3>
The ZS-DeconvNet Fiji plugin was developed based on TensorFlow-Java 1.15.0, which is compatible with CUDA version of 10.1 and cuDNN version of 7.5.1. If you would like to process models with a different TensorFlow version, or running with different GPU settings, please do the following:

+ Open <i>Edit > Options > Tensorflow</i>, and choose the version matching your model or setting.
+ Wait until a message pops up telling you that the library was installed.
+ Restart Fiji.

<h3 style="color:white;">5.3 Inference with ZS-DeconvNet Fiji plugin</h3>

Given a pre-trained ZS-DeconvNet model and an image or stack to be processed, the Fiji plugin is able to generate the corresponding denoised (optional) and super-resolved deconvolution image or stack. The workflow includes following steps: 

+ <p>Open the image or stack in Fiji and start ZS-DeconvNet plugin by Clicking <i>Plugins > ZS-DeconvNet > predict ZS-DeconvNet 2D / predict ZS-DeconvNet 3D</i>.</p>
+ <p>Select the network model file, i.e., .zip file in the format of BioImage Model Zoo bundle. Of note, the model file could be trained and saved either by Python codes (see <a href='https://gist.github.com/asimshankar/000b8d276f211f972168afa138eb3cc7'>this gist</a>) or ZS-DeconvNet Fiji plugin, but has to be saved with TensorFlow environment <= 1.15.0.</p>
+ <p>Check inference options and choose hyper-parameters used in the inference. The options and parameters here are primarily selected to properly normalize the input data (NormalizeInput, PercentileBottom, and PerventileTop), perform tiling prediction to save memory of CPUs or GPUs (Number of tiles, Overlap between tiles, and Batch size), and decide whether to show progress dialog and denoising results or not (Show progress dialog and Show denoising result). See the ReadMe.md for detailed parameter table. </p>
+ <p>After image processing with status bar shown in the message box (if select Show process dialog), the denoised (if select Show denoising result) and deconvolved output will pop out in separate Fiji windows automatically. Then the processed images or stacks could be viewed, manipulated, and saved via Fiji.</p>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet-page/blob/page/images/SuppFig17_Fiji_Plugin_v2_white_logo.png?raw=true" width="900" align="middle" /></center>

<h3 style="color:white;">5.4 Training with ZS-DeconvNet Fiji plugin</h3>

<p>For ZS-DeconvNet model training, we generally provide two commands: <i>train on augmented data</i> and <i>train on opened img</i>, which differ in the ways of data loading and augmentation. The former command loads input data and corresponding GT images which are augmented elsewhere, e.g., in MATLAB or Python, from two data folders file by file, and the latter command directly takes the image stack opened in the current Fiji window as the training data and automatically perform data augmentation including image re-corruption (for 2D cases), random cropping, rotation and flipping into a pre-specified patch number. </p>

The overall workflow of ZS-DeconvNet training with Fiji plugin includes following steps:

+ <p>Open the image or stack to be used for training in Fiji and start the ZS-DeconvNet plugin by clicking <i>Plugins > ZS-DeconvNet > train on opened img</i>; or directly start the plugin by the alternative command <i>Plugins > ZS-DeconvNet > train on augmented data</i> and select the folders containing input images and GT images.</p>
+ <p>Select the network type, i.e., 2D ZS-DeconvNet or 3D ZS-DeconvNet, the PSF file used for calculating deconvolution loss and choose training hyper-parameters, which include total epochs, iteration number per epoch, batch size, and initial learning rate. For 2D ZS-DeconvNet training by the command of <i>train on opened img</i>, three extra recorruption-related parameters of $\alpha $, $\beta _1$, and $\beta _2$ are tuneable, where $\alpha $ and $\beta _1$ are set as [1, 2] and [0.5, 1.5] by default, and $\beta _2$ should be set as the standard deviation of the camera background, which could be pre-calibrated from blank frames or calculated from empty regions of the training data. See the ReadMe.md for detailed parameter table.</p>
+ <p>Click OK to start training. A message box containing training information will pop up, and three preview windows will be displayed after each epoch, showing the current input images, denoised output images and deconvolution output images. </p>
+ Three types of exit:<br>
(i) Press <i>Cancel > Close</i> to enforce an exit if you don't want to train or save this model.<br>
(ii) Press <i>Finish Training</i> for an early stopping. A window will pop up and you can save the model by <i>File actions > Save to..</i>.<br>
(iii) After the training is completed, a window will pop up and you can save the model by <i>File actions > Save to..</i>.
  
  Of note, you can also press <i>Export Model</i> during training to export the lastest model without disposing the training progress.

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet-page/blob/page/images/SuppFig16_Fiji_Plugin_Training_v1_whiteBG.png?raw=true" width="900" align="middle" /></center>