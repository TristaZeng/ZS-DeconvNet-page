---
layout: page
title: Tutorial

---

<br>
<center><img src="https://github.com/TristaZeng/ZS-DeconvNet-page/blob/page/images/Logo_v2_White_transparent.png?raw=true" width="500" align="center" /></center>

<p>Our source code and Fiji plugin can be downloaded from <a href='https://github.com/TristaZeng/ZS-DeconvNet'>https://github.com/TristaZeng/ZS-DeconvNet</a>. We use MATLAB R2021b to generate training datasets from raw images, and TensorFlow to perform training and inference. We also develop a Fiji plugin for both data generation and model training and inference.</p>
<p>This page provides <b>quick-start tutorials</b> for MATLAB, Python and Fiji. For more detailed instructions see the ReadMe.md in our <a href='https://github.com/TristaZeng/ZS-DeconvNet'>Github repository</a>. An overlook of their respective functions are summarized in the table below:</p>

| Environment                    | Data augmentation | Network training| 
  |:----------------------------------:|:-------------:|:-------------:|
  | MATLAB                    | 1. Save augmented images in local folders<br>2. Convenient for visualizing and checking the augmented data| / | 
  | Python                    | / | 1. Faster than Fiji by ~1.6-fold<br>2. Automatically save trained models | 
  | Fiji plugin                    | 1. Simple operation<br>2. Save augmented images in memory and directly train new models with them| 1. Simple operation<br>2. Automatically show the intermediate results<br>3. Save the models manually during the training process| 



<h2 style="color:white;">Content</h2>

<ul>
  <li><a href="#params">Brief description of parameters</a></li>
  <li><a href="#Data Pre-processing">MATLAB: How to generate training dataset and PSFs</a></li>
  <li><a href="#Implementation of Python code">Python: How to perform training and inference</a></li>
  <li><a href="#Fiji plugin">Fiji: How to do all above with a click</a></li>
</ul>

<hr>

<h2 style="color:white;" id="params">Brief description of parameters</h2>

<p>ZS-DeconvNet is a deep-learning method for denoising and super-resolution with no need for users to traverse through hyper-parameters so you can skip this part without inhibiting the using of ZS-DeconvNet. Apart from the typical hyper-parameters such as learning rate, total epoch, batch size and patch size, there are a few important inputs:</p>

| Name                                             | Suggested value       | Description                                                                                              |
|:-----------------------------------------------------------:|:-------------------:|:--------------------------------------------------------------------------------------------------------:|
| Alpha for recorruption   | 1-2 | Theorectically any value is applicable. Default value [1,2] is used for all simulated and experimental dataset of various specimens and imaging conditions in this paper.              |
| Beta1 for recorruption  | 0.5-1.5 | The theoretically optimal value is 1. Default value [0.5, 1.5] is used for all simulated and experimental dataset of various specimens and imaging conditions in this paper.              |
| Beta2 for recorruption   | estimated from data | The variance of the readout noise of the camera, which can be estimated from the sample-free region of the images in training dataset or pre-calibrated from the camera following standard protocols.               |
| PSF file                                                    |    /                 | Root path of the point spread function file used for calculating deconvolution loss. The PSF size has to be an odd number. We included the corresponding PSF for each type of data in our [open-source datasets](https://www.zenodo.org/record/7261163#.ZD9kZHZBx3g). |
| Weights of Hessian Reg.                                     | 0.02 (2D) / 0.1 (3D)                 | The scalar weight to balance the Hessian regularization in the loss function. Default values are used for all simulated and experimental dataset of various specimens and imaging conditions in this paper.                                                               |

<hr>

<h2 style="color:white;" id="Data pre-processing">MATLAB: How to generate training dataset and PSFs</h2>

<p>We use MATLAB R2021b but other versions might be compatible. </p>

<b>1.1</b> To generate training dataset, you can:

+ <p>Prepare a folder of raw data. Download <a href='https://www.zenodo.org/record/7261163#.ZD9kZHZBx3g'>our open-source raw data</a> of various modalities or use your own raw data.</p> 

+ <p>Open <code style="background-color:#393939;">./Python_MATLAB_Codes/data_augment_recorrupt_matlab/GenData4ZS-DeconvNet/demo_augm.m</code> and replace the parameter <code style="background-color:#393939;">data_folder</code> with your raw data directory.</p>

+ <p>The default output path is <code style="background-color:#393939;">./Python_MATLAB_Codes/data_augment_recorrupt_matlab/your_augmented_datasets/</code>.</p>

<b>1.2</b> To generate training dataset for ZS-DeconvNet-SIM, you can:

+ <p>Prepare a folder of raw data.</p> 

+ <p>Open <code style="background-color:#393939;">./Python_MATLAB_Codes/data_augment_recorrupt_matlab/GenData4ZS-DeconvNet-SIM/Create_corrupt_img_2D.m</code> or <code style="background-color:#393939;">Create_corrupt_img_3D.m</code> and set the parameters.</p>

+ <p>Perform SIM reconstructions on the corrupted dataset to generate the input and targets for ZS-DeconvNet-SIM.</p>

<b>2.1</b> To simulate PSF for (3D) ZS-DeconvNet, run <code style="background-color:#393939;">./Python_MATLAB_Codes/data_augment_recorrupt_matlab/GenData4ZS-DeconvNet/create_PSF.m</code> and the simulated PSF will be saved to your designated file path.

<b>2.2</b> To simulate PSF for ZS-DeconvNet-SIM, run <code style="background-color:#393939;">./Python_MATLAB_Codes/data_augment_recorrupt_matlab/GenData4ZS-DeconvNet-SIM/main_create_simu_beads.m</code> and the generated raw 3D-SIM images of a simulated bead <code style="background-color:#393939;">img_sim</code> will be saved to your MATLAB workspace.

<hr>

<h2 style="color:white;" id="Implementation of Python code">Python: How to perform training and inference</h2>

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

Skip this part if you do not wish to train a new model. You can just test <a href='https://drive.google.com/drive/folders/1XAOuLYXYFCxlElRwvik_fs7TqZlRixGv'>the demo test data using our provided pre-trained models</a>. 

To train a new model, you need to:

+ Generate the training dataset following the instructions in the previous part.
+ Choose a test image/volume and obtain the path to the corresponding PSF.
+ Change <code style="background-color:#393939;">otf_or_psf_path</code> (or <code style="background-color:#393939;">psf_path</code> in the case of 3D), <code style="background-color:#393939;">data_dir</code>, <code style="background-color:#393939;">folder</code> and <code style="background-color:#393939;">test_images_path</code> in <code style="background-color:#393939;">./Python_MATLAB_Codes/train_inference_python/train_demo_2D.sh</code> or <code style="background-color:#393939;">train_demo_3D.sh</code>.
+ Run it in your terminal.
+ The result wills be saved to <code style="background-color:#393939;">./Python_MATLAB_Codes/your_saved_models/</code>.
+ Run <code style="background-color:#393939;">tensorboard --logdir [save_weights_dir]/[save_weights_name]/graph</code> to monitor the training process via tensorboard if needed.
+ Other <b>detailed description of each input argument of the python codes</b> can be found in the comments of <code style="background-color:#393939;">./Python_MATLAB_Codes/train_inference_python/train_demo_2D.sh</code> or <code style="background-color:#393939;">train_demo_3D.sh</code>.

<h3 style="color:white;">4.3 Inference Demo</h3>

We have provided a <a href='https://drive.google.com/drive/folders/1XAOuLYXYFCxlElRwvik_fs7TqZlRixGv'>download link</a> in the folder <code style="background-color:#393939;">./Python_MATLAB_Codes/saved_models/</code>, which contains six different data samples, their corresponding well-trained ZS-DeconvNet models, and their ZS-DeconvNet outputs, organized in the format of:

+ <code style="background-color:#393939;">./Python_MATLAB_Codes/saved_models/xx/test_data</code> contains the test data of this sample type.
+ <code style="background-color:#393939;">./Python_MATLAB_Codes/saved_models/xx/saved_model</code> contains the pre-trained model (a .h5 file) of this sample type.
+ <code style="background-color:#393939;">./Python_MATLAB_Codes/saved_models/xx/saved_model/Inference_demo</code> contains the ZS-DeconvNet output of this sample type. You should be able to get the same result if you use the pre-trained model to process the test data in <code style="background-color:#393939;">./Python_MATLAB_Codes/saved_models/xx/test_data</code>.

Now to test a well-trained ZS-DeconvNet model, you should:

+ Change the weight paths in <code style="background-color:#393939;">./Python_MATLAB_Codes/train_inference_python/infer_demo_2D.sh</code> or <code style="background-color:#393939;">infer_demo_3D.sh</code> accordingly, or just use the default options given by us. 
+ Run it in your terminal.
+ The output will be saved to the folder where you load weights, e.g., if you load weights from <code style="background-color:#393939;">./Python_MATLAB_Codes/train_inference_python/saved_models/.../weights_40000.h5</code>, then the output will be saved to <code style="background-color:#393939;">./Python_MATLAB_Codes/train_inference_python/saved_models/.../Inference/</code>.

<hr>

<h2 style="color:white;" id="Fiji plugin">Fiji: How to do all above with a click</h2>

We provide a simple instruction for our plugin here. For latest updates, detailed parameter table and snapshots of usage, see the ReadMe.md in <a href='https://github.com/TristaZeng/ZS-DeconvNet/tree/main/Fiji_Plugin'>our Github repository</a>.

Here is a demo video of screen captures of the training and inference procedures of the ZS-DeconvNet Fiji plugin on two different low SNR dataset with two types of operating systems (Windows 10 and Ubuntu 18.04.6 LTS):

<center><video src="https://github.com/TristaZeng/ZS-DeconvNet-page/blob/page/video/SuppVideo9_Demo_Plugin.mp4?raw=true" controls="controls" width="100%" height="auto"/></center>

<h3 style="color:white;">5.1 Installation and Pre-trained Models</h3>
<p>You can follow the instructions below to install the plugin:</p>

+ <p>Download from <a href='https://drive.google.com/drive/folders/1nJoj9Ljx2MNXa-lCOGIzVj_1BT-xrp2F'>here</a>.</p>
+ <p>Copy <code style="background-color:#393939;">./jars/*</code> and <code style="background-color:#393939;">./plugins/*</code> to your root path of Fiji <code style="background-color:#393939;">/*/Fiji.app/</code>.</p>
+ <p>Restart Fiji.</p>
+ <p>We provide pre-trained models in BioImage Model Zoo bundle and one corresponding test image or test stack for each model in the folder <code style="background-color:#393939;">./pre-trained_models</code>. See the list below:</p>

| Model Name     |    Model Type |        Test Data Name | Test Data Type |
  |:---------------------------:|:-------------:|:-------------:|:-------------:|
  | ZS-DeconvNet-2D-WF-lamp1.zip | ZS-DeconvNet | ZS-DeconvNet-2D-WF-lamp1-input.tif | 2D WF |
  | ZS-DeconvNet-2D-WF-ER.zip | ZS-DeconvNet | ZS-DeconvNet-2D-WF-ER-input.tif | 2D WF |
  | ZS-DeconvNet-3D-LLSM-Mitochondria.zip  | 3D ZS-DeconvNet | ZS-DeconvNet-3D-LLSM-Mitochondria-input.tif | 3D LLSM |
  | ZS-DeconvNet-3D-Confocal-MT.zip | 3D ZS-DeconvNet | ZS-DeconvNet-3D-Confocal-MT-input.tif | 3D Confocal Microscopy |
  | ZS-DeconvNet-2D-SIM-MT.zip | ZS-DeconvNet-SIM | ZS-DeconvNet-2D-SIM-MT-input.tif | 2D SIM |
  | ZS-DeconvNet-2D-SIM-CCPs.zip | ZS-DeconvNet-SIM | ZS-DeconvNet-2D-SIM-CCPs-input.tif | 2D SIM |

We mainly developed and tested the ZS-DeconvNet Fiji plugin on workstations of <b>Linux and Windows</b> operating system equipped with Nvidia graphics cards. Because TensorFlow-GPU package is currently incompatible with MacOS, we are sorry that MacBook users can only use the TensorFlow-CPU to run our ZS-DeconvNet Fiji plugin at present, which is relatively inefficient compared to Nvidia GPU-based computation. We’ll be looking for the solutions and trying to make our plugin compatible with MacBook for higher efficiency in the future.

<h3 style="color:white;">5.2 About GPU and TensorFlow version</h3>
The ZS-DeconvNet Fiji plugin was developed based on TensorFlow-Java 1.15.0, which is compatible with CUDA version of 10.1 and cuDNN version of 7.5.1. If you would like to process models with a different TensorFlow version, or running with different GPU settings, please do the following:

+ Open <i>Edit > Options > Tensorflow</i>, and choose the version matching your model or setting.
+ Wait until a message pops up telling you that the library was installed.
+ Restart Fiji.

<h3 style="color:white;">5.3 Inference with ZS-DeconvNet Fiji plugin</h3>

Given a pre-trained ZS-DeconvNet model and an image or stack to be processed, the Fiji plugin is able to generate the corresponding denoised (optional) and super-resolved deconvolution image or stack. The workflow includes following steps: 

+ <p>Open the image or stack in Fiji and start ZS-DeconvNet plugin by Clicking <i>Plugins > ZS-DeconvNet > predict ZS-DeconvNet 2D / predict ZS-DeconvNet 3D</i>.</p>
+ <p>Select the network model file, i.e., .zip file in the format of BioImage Model Zoo bundle. Of note, the model file could be trained and saved either by Python codes (see <a href='https://gist.github.com/asimshankar/000b8d276f211f972168afa138eb3cc7'>this gist</a>) or ZS-DeconvNet Fiji plugin, but has to be saved with TensorFlow environment <= 1.15.0.</p>
+ <p>Adjust inference options if you like. </p>
+ <p>After image processing with status bar shown in the message box (if select Show process dialog), the denoised (if select Show denoising result) and deconvolved output will pop out in separate Fiji windows automatically. Then the processed images or stacks could be viewed, manipulated, and saved via Fiji.</p>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet-page/blob/page/images/SuppFig17_Fiji_Plugin_v2_white_logo.png?raw=true" width="900" align="middle" /></center>

<h3 style="color:white;">5.4 Training with ZS-DeconvNet Fiji plugin</h3>

<p>For ZS-DeconvNet model training, we generally provide two commands: <i>train on augmented data</i> and <i>train on opened img</i>, which differ in the ways of data loading and augmentation. The former command loads input data and corresponding GT images which are augmented elsewhere, e.g., in MATLAB or Python, from two data folders file by file, and the latter command directly takes the image stack opened in the current Fiji window as the training data and automatically perform data augmentation. </p>

The overall workflow of ZS-DeconvNet training with Fiji plugin includes following steps:

+ <p>Open the image or stack to be used for training in Fiji and start the ZS-DeconvNet plugin by clicking <i>Plugins > ZS-DeconvNet > train on opened img</i>; or directly start the plugin by the alternative command <i>Plugins > ZS-DeconvNet > train on augmented data</i> and select the folders containing input images and GT images.</p>
+ <p>Select the network type, i.e., 2D ZS-DeconvNet or 3D ZS-DeconvNet and the PSF file.</p>
+ <p>Adjust other options if you like, or do not.</p>
+ <p>Click OK to start training. A message box containing training information will pop up, and three preview windows will be displayed after each epoch, showing the current input images, denoised output images and deconvolution output images. </p>
+ Three types of exit:<br>
(i) Press <i>Cancel > Close</i> to enforce an exit if you don't want to train or save this model.<br>
(ii) Press <i>Finish Training</i> for an early stopping. A window will pop up and you can save the model by <i>File actions > Save to..</i>.<br>
(iii) After the training is completed, a window will pop up and you can save the model by <i>File actions > Save to..</i>.
  
  Of note, you can also press <i>Export Model</i> during training to export the lastest model without disposing the training progress.

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet-page/blob/page/images/SuppFig20_Fiji_Plugin_Training_v2_whiteBG.png?raw=true" width="900" align="middle" /></center>

<h3 style="color:white;">5.5 PSF Generation</h3>

<p>You can generate PSF via the Fiji plugin PSF Generator.</p>