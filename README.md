# LCR
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![repo size](https://img.shields.io/github/repo-size/xinychen/LCR.svg)](https://github.com/xinychen/LCR/archive/master.zip)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/LCR.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/LCR)

<h6 align="center">Made by Xinyu Chen • :globe_with_meridians: <a href="https://xinychen.github.io">https://xinychen.github.io</a></h6>

<br>

> Laplacian convolutional representation for traffic time series imputation.

<br>

## Documentation

### Problem Definition

In this research, we aim at addressing traffic time series imputation problems by carefully utilizing global and local trends underlying the data, see Figure 1. While the global time series trend can be modeled by the circulant matrix nuclear norm optimization, we propose to characterize the local time series trend with Laplacian kernels and regularization. 

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/pickup_dropoff_trips_nyc_2024_april_may.png" width="700" />
</p>

<p align = "center">
<b>Figure 1.</b> Traffic time series imputation on missing values.
</p>


<br>

## Features

LCR is an efficient algorithm for univariate and multivariate time series imputation. In this repository, we aim to support you to explore traffic time series imputation problems with global and local trend modeling, referring to circulant matrix nuclear norm and Laplacian regularization, respectively. We provide friendly implementation with a few lines relying on the `numpy` package in Python.

<br>

## Cite Us

- Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). [Laplacian convolutional representation for traffic time series imputation](https://doi.org/10.1109/TKDE.2024.3419698). IEEE Transactions on Knowledge and Data Engineering. Early Access. [[PDF](https://xinychen.github.io/papers/Laplacian_convolution.pdf)] [[Slides](https://xinychen.github.io/slides/LCR24.pdf)]

or 

```tex
@article{chen2024laplacian,
  title={Laplacian convolutional representation for traffic time series imputation},
  author={Chen, Xinyu and Cheng, Zhanhong and Cai, HanQin and Saunier, Nicolas and Sun, Lijun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```

<br>

## Supported by

<a href="https://ivado.ca/en">
<img align="middle" src="graphics/ivado_logo.jpeg" alt="drawing" height="70" hspace="50">
</a>
<a href="https://www.cirrelt.ca/">
<img align="middle" src="graphics/cirrelt_logo.png" alt="drawing" height="50">
</a>
