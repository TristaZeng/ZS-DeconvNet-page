---
layout: page
title: Principle

---

<br>
<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/Logo_v2_White_transparent.png?raw=true" width="500" align="center" /></center>

<h2 style="color:white;">Content</h2>
<ul>
  <li><a href="#Theory">Theory</a></li>
  <li><a href="#Schematic">Schematic</a></li>
</ul>

<h2 style="color:white;" id="Theory">Theory</h2>

The concept of ZS-DeconvNet is based on the optical imaging forward model informed unsupervised inverse problem solver:

$$
argmin_\theta \quad ||y-(H×f_\theta (y))_\downarrow||_2^2 \tag{1}
$$

where y denotes the noisy LR image, H denotes the points spread function (PSF) matrix, and $f_\theta (y)$ is the deep neural networks (DNNs) with
trainable parameters $\theta$. × and $\downarrow$ indicates matrix multiplication and
down-sampling operators.

However, DNNs trained directly via the above objective function enhance both useful information of samples and useless noise induced from the acquisition process, e.g., shooting noise, thus their performance degrades rapidly as the SNR of input images declines. To equip ZS-DeconvNet with robustness to
noises while maintaining its unsupervised characteristic, we adopted image
de-noising schemes and classify ZS-DeconvNet cases as follows:

<h3 style="color:white;">2D ZS-DeconvNet</h3>

The image pairs $(\hat{y},\tilde{y})$ used for training 2D ZS-DeconvNet models were generated following a modified scheme from the original re-corrupted to re-corrupted strategy[1] under the assumption of mixed Poisson-Gaussian noise distributions, where three hyper parameters $\beta_1,\beta_2, \alpha$ needed to be pre-characterized. The re-corruption procedure from a single noisy image y can be represented in matrix form as:

$$
\hat{y}=y+D^Tz,
\tilde{y}=y-D^{-1}z \tag{2}
$$

where $D=\alpha I$ is an invertible matrix defined as a magnified unit matrix by a factor of α, which controls the overall magnitude of added noises, and z is a random noise map sampled from a Gaussian distribution with zero means:

$$
z \sim N(0,\sigma ^2I),
\sigma ^2 = \beta_1H(y-b)+\beta_2 \tag{3}
$$

where $\beta_1$ is the Poissonian factor affecting the variance of the signal-dependent shot noise, and $\beta_2$ is the Gaussian factor representing the variance of additive Gaussian noises. b is the background, approximately regarded as a fixed value related to the camera, by subtracting which we extracted fluorescence signals from the sample. H(∙) is a linear low-pass filter used to preliminarily smooth the image and reduce the noise, and we adopted an averaging filter with a size of 5 pixels in our experiments.

In practice, we use rotation, flipping and cropping to get patches of specified size from the raw data, and implement the above re-corruption process to each patch to generate re-corrupted pairs.

We designed a combined loss function consisting of a denoising term and a deconvolution term, which respectively corresponds to the denoising stage and the deconvolution stage:

$$
L(\hat{y},\tilde{y})=\mu L_{den}(\hat{y},\tilde{y})+(1-\mu)L_{dec}(\hat{y},\tilde{y}) \tag{4}
$$

where $(\hat{y},\tilde{y})$ indicates the re-corrupted image pair mentioned above, and $\mu$ is a scalar weighting factor to
balance the two terms, which we empirically set as 0.5 in our experiments. The
denoising loss $L_{den}$ and the deconvolution loss $L_{dec}$ are defined as follows:

$$
L_{den}(\hat{y},\tilde{y})=||f_{\theta '}(\hat{y})-\tilde{y}||_2^2 \tag{5}
$$

$$
L_{dec}(\hat{y},\tilde{y})=||(f_\theta (\hat{y})*PSF)_\downarrow -\tilde{y}||_2^2+\lambda R_{Hessian}(f_\theta (\hat{y})) \tag{6}
$$

where $f_{\theta '}(\hat{y}),f_\theta (\hat{y})$  are the output images of the denoising stage and the deconvolution stage, $R_{Hessian}$ is the Hessian regularization term used to
regulating the solution space, and $\lambda$ is the weighting scalar to balance the impact of the regularization.

<h3 style="color:white;">3D ZS-DeconvNet</h3>

Similar to the 2D case, we designed a combined loss function consisting of a denoising term and a deconvolution term:

$$
L(z)=\mu L_{den}(z)+(1-\mu)L_{dec}(z) \tag{7}
$$

where z is the 3D noisy image stack, and denoising loss $L_{den}$ and the deconvolution loss $L_{dec}$ are defined as follows:

$$
L_{den}(z)=||f_{\theta '}(S_{odd}(z))-S_{even}(z)||_2+\gamma ||f_{\theta '}(S_{odd}(z))-S_{even}(z)-(S_{odd}(f_{\theta '}(z))-S_{even}(f_{\theta '}(z)))||_2 \tag{8}
$$

$$
L_{dec}(z)=||f_\theta (S_{odd}(z))*PSF-S_{even}(z)||_2+\gamma ||(f_\theta (S_{odd}(z))*PSF)_\downarrow -S_{even}(z)-(S_{odd}(f_{\theta '}(z))-S_{even}(f_{\theta '}(z)))||_2 \\ +\lambda R_{Hessian}(f_\theta (S_{odd}(z))) \tag{9}
$$

where $S_{odd}$ and $S_{even}$ represent the axial sampling operators which takes an image stack and returns its odd slices or even slices, respectively, stacked
in the same order as the original stack, $\gamma$  and $\lambda$ are weighting scalars of the GAR term and the Hessian regularization term.

<h3 style="color:white;">2D ZS-DeconvNet-SIM</h3>

We have proven in our Supplementary Note 1c that the SIM reconstruction noise is of zero mean. This zero-mean characteristics of reconstruction artifacts make it possible to perform denoising and deconvolution for SIM images in an unsupervised manner. In practical implementation of 2D ZS-DeconvNet-SIM, we first added additional noises for each raw SIM images of different orientations and phases, i.e., 3-orientation × 3-phase, via Eq. (2) to generate two sets of recorrupted raw SIM images, and then the generated images were reconstructed into two noisy SR-SIM images, denoted as $\hat{Y}$ and $\tilde{Y}$, which were used as the input and GT in the training procedure.

For the dual-stage architecture of ZS-DeconvNet-SIM, we set its overall loss function of the same form with Eq. (4), and the denoising loss is calculated with the two recorrupted SIM images:

$$
L_{den} (\hat{Y},\tilde{Y})=‖f_{\theta '} (\hat{Y})-\tilde{Y}‖_2^2,\tag{10}
$$

where $f_{\theta '}$ is the denoising stage of ZS-DeconvNet-SIM with corresponding trainable parameters $\theta '$. 

Similar to ZS-DeconvNet for acquired raw image processing, we next defined tha deconvolution loss for ZS-DeconvNet-SIM based on recorrupted SIM image pairs and the super-resolution PSF matrix $H_{SIM}$ as

$$
L_{dec} (\hat{Y},\tilde{Y})=‖H_{SIM} f_\theta (\hat{Y})-\tilde{Y}‖_2^2+\lambda R_{Hessian} (f_\theta (\hat{Y})) ),\tag{11}
$$

where $f_\theta $ is the entire dual-stage network with all trainable parameters $\theta $.

<h3 style="color:white;">3D ZS-DeconvNet-SIM</h3>

The applications of 3D ZS-DeconvNet-SIM for volumetric SIM modalities such as lattice light-sheet structured illumination microscopy (LLS-SIM) and three-dimensional structured illumination microscopy (3D-SIM) are similar to those of 3D ZS-DeconvNet described in Supplementary Note 1b with the primary difference being that 3D ZS-DeconvNet-SIM adopts spatially interleaved post-reconstructed SIM images rather than noisy raw images as inputs and GT in both training and inference phases. The objective function of 3D ZS-DeconvNet-SIM is devised as the combination of the denoising loss and the deconvolution loss, which is formulated as follows

$$
L(Z)= \mu L_{den} (Z)+(1-\mu ) L_{dec} (Z),\tag{12}
$$

$$
L_{den} (Z) =⁡‖f_{\theta ' } (\hat{Z})-\tilde{Z}‖_2^2+\gamma ‖f_{\theta ' } (\hat{Z})-\tilde{Z}-(S_{odd} (f_{\theta ' } (Z))-S_{even} (f_{\theta ' } (Z)))‖_2^2,\tag{13}
$$

$$
L_{dec} (Z)=‖H_{SIM} f_{\theta} (\hat{Z})-\tilde{Z}‖_2^2+\gamma ‖H_{SIM} f_{\theta } (\hat{Z})-\tilde{Z}-S_{odd} (f_{\theta ' } (Z))+S_{even} (f_{\theta ' } (Z))‖_2^2+\lambda R_{Hessian} (f_{\theta }(\hat{Z})),\tag{14}
$$

where $Z$, $\hat{Z}$, and $\tilde{Z}$ are the entire stack, odd slices, and even slices of the noisy SIM image stack generated via the analytical SIM reconstruction algorithm, $H_{SIM}$ is the volumetric PSF of corresponding SIM systems. 

<h2 style="color:white;" id="Schematic">Schematic</h2>

<center><h3 style="color:white;">Schematic of 2D ZS-DeconvNet</h3></center>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/R2R.png?raw=true" width="850" align="middle"></center>

<center><h3 style="color:white;">Schematic of 3D ZS-DeconvNet</h3></center>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/NBR2NBR.png?raw=true" width="850" align="middle"></center>

<center><h3 style="color:white;">Schematic of 2D ZS-DeconvNet-SIM</h3></center>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/R2R_SIM.png?raw=true" width="850" align="middle"></center>

<center><h3 style="color:white;">Schematic of 3D ZS-DeconvNet-SIM</h3></center>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/NBR2NBR_SIM.png?raw=true" width="850" align="middle"></center>
