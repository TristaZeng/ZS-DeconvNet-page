---
layout: page
title: Principle

---

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/Logo_v2_White_transparent.png?raw=true" width="500" align="center" /></center>

<h2 style="color:white;">Content</h2>
<ul>
  <li><a href="#Theory">Theory</a></li>
  <li><a href="#Schematic">Schematic</a></li>
</ul>

<h2 style="color:white;" id="Theory">Theory</h2>

The concept of ZS-DeconvNet is based on the optical imaging forward model informed unsupervised inverse problem solver:

$$
argmin_\theta \quad ||y-(H×f_\theta (y))_\downarrow||_2^2
$$

where y denotes the noisy LR image, H denotes the points spread function (PSF) matrix, and $f_\theta (y)$ is the deep neural networks (DNNs) with
trainable parameters $\theta$. × and $\downarrow$ indicates matrix multiplication and
down-sampling operators.

However, DNNs trained directly via the above objective function enhance both useful information of samples and useless noise induced from the acquisition process, e.g., shooting noise, thus their performance degrades rapidly as the SNR of input images declines. To equip ZS-DeconvNet with robustness to
noises while maintaining its unsupervised characteristic, we adopted image
de-noising schemes as follows:

<h3 style="color:white;">R2R scheme for 2D images</h3>

We designed a combined loss function consisting of a denoising term and a deconvolution term, which respectively corresponds to the denoising stage and the deconvolution stage:

$$
L(\hat{y},\tilde{y})=\mu L_{den}(\hat{y},\tilde{y})+(1-\mu)L_{dec}(\hat{y},\tilde{y})
$$

where $(\hat{y},\tilde{y})$ indicates the re-corrupted image pair (see Principle for detailed implementation of re-corruption), and $\mu$ is a scalar weighting factor to
balance the two terms, which we empirically set as 0.5 in our experiments. The
denoising loss $L_{den}$ and the deconvolution loss $L_{dec}$ are defined as follows:

$$
L_{den}(\hat{y},\tilde{y})=||f_{\theta '}(\hat{y})-\tilde{y}||_2^2
$$

$$
L_{dec}(\hat{y},\tilde{y})=||(f_\theta (\hat{y})*PSF)_\downarrow -\tilde{y}||_2^2+\lambda R_{Hessian}(f_\theta (\hat{y}))
$$

where $f_{\theta '}(\hat{y}),f_\theta (\hat{y})$  are the output images of the denoising stage and the deconvolution stage, $R_{Hessian}$ is the Hessian regularization term used to
regulating the solution space, and $\lambda$ is the weighting scalar to balance the impact of the regularization.

<h3 style="color:white;">SiS scheme for 3D stacks</h3>

Similar to the 2D case, we designed a combined loss function consisting of a denoising term and a deconvolution term:

$$
L(z)=\mu L_{den}(z)+(1-\mu)L_{dec}(z)
$$

where z is the 3D noisy image stack, and denoising loss $L_{den}$ and the deconvolution loss $L_{dec}$ are defined as follows:

$$
L_{den}(z)=||f_{\theta '}(S_{odd}(z))-S_{even}(z)||_2+\gamma ||f_{\theta '}(S_{odd}(z))-S_{even}(z)-(S_{odd}(f_{\theta '}(z))-S_{even}(f_{\theta '}(z)))||_2
$$

$$
L_{dec}(z)=||f_\theta (S_{odd}(z))*PSF-S_{even}(z)||_2+\gamma ||(f_\theta (S_{odd}(z))*PSF)_\downarrow -S_{even}(z)-(S_{odd}(f_{\theta '}(z))-S_{even}(f_{\theta '}(z)))||_2+\lambda R_{Hessian}(f_\theta (S_{odd}(z)))
$$

where $S_{odd}$ and $S_{even}$ represent the axial sampling operators which takes an image stack and returns its odd slices or even slices, respectively, stacked
in the same order as the original stack, $\gamma$  and $\lambda$ are weighting scalars of the GAR term and the Hessian regularization term.

<h3 style="color:white;">For SIM Deconv.</h3>

<h2 style="color:white;" id="Schematic">Schematic</h2>

<center><h3 style="color:white;">Schematic of ZS-DeconvNet</h3></center>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/R2R.png?raw=true" width="850" align="middle"></center>

<center><h3 style="color:white;">Schematic of ZS-DeconvNet (3D)</h3></center>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/NBR2NBR.png?raw=true" width="850" align="middle"></center>

<center><h3 style="color:white;">Schematic of ZS-DeconvNet (SIM)</h3></center>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/R2R_SIM.png?raw=true" width="850" align="middle"></center>

<center><h3 style="color:white;">Schematic of ZS-DeconvNet (3D SIM)</h3></center>

<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/NBR2NBR_SIM.png?raw=true" width="850" align="middle"></center>
