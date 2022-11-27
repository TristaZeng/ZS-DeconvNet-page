---
layout: page
title: ZS-DeconvNet tutorial

---
<br>
<center><img src="https://github.com/TristaZeng/ZS-DeconvNet/blob/master/images/Logo_v2_White_transparent.png?raw=true" width="500" align="center" /></center>

<h2 style="color:white;">Content</h2>

<ul>
  <li><a href="#Data Pre-processing">Data pre-processing</a></li>
  <li><a href="#Implementation of Python code">Implementation of Python code</a></li>
  <li><a href="#Fiji plugin">Fiji plugin</a></li>
</ul>

<h2 style="color:white;" id="Data pre-processing">1. Data pre-processing</h2>

<h3 style="color:white;">1.1 Data Augmentation and Re-corruption for 2D Data</h3>

The image pairs $(\hat{y},\tilde{y})$ used for training 2D ZS-DeconvNet models were generated following a modified scheme from the original re-corrupted to re-corrupted strategy[1] under the assumption of mixed Poisson-Gaussian noise distributions, where three hyper parameters $\beta_1,\beta_2, \alpha$ were needed to be pre-characterized. The re-corruption procedure from a single noisy image y can be represented in matrix form as:

$$
\hat{y}=y+D^Tz
$$

$$
\tilde{y}=y-D^{-1}z
$$

where $D=\alpha I$ is an invertible matrix defined as a magnified unit matrix by a factor of α, which controls the overall magnitude of added noises, and z is a random noise map sampled from a Gaussian distribution with zero means:

$$
z \sim N(0,\sigma ^2I)
$$

$$
\sigma ^2 = \beta_1H(y-b)+\beta_2
$$

where $\beta_1$ is the Poissonian factor affecting the variance of the signal-dependent shot noise, and $\beta_2$ is the Gaussian factor representing the variance of additive Gaussian noises. b is the background, approximately regarded as a fixed value related to the camera, by subtracting which we extracted fluorescence signals from the sample. H(∙) is a linear low-pass filter used to preliminarily smooth the image and reduce the noise, and we adopted an averaging filter with a size of 5 pixels in our experiments.

<h3 style="color:white;">1.2 Data Augmentation and Re-sampling for 3D Data</h3>


<h2 style="color:white;" id="Implementation of Python code">2. Implementation of Python code</h2>

<h2 style="color:white;" id="Fiji plugin">3. Fiji plugin</h2>
