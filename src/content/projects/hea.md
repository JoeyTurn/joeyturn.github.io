---
title: Rundown of "Predicting Kernel Regression Learning Curves from only Raw Data Statistics" - the HEA
pubDate: "2025-10-11"
description: "On real data, we know exactly how kernel regression performs."
links:
  - { label: "arXiv",  url: "http://arxiv.org/abs/2510.14878" }
  - { label: "GitHub", url: "https://github.com/JoeyTurn/hermite-eigenstructure-ansatz" }
authors:
  - { name: "Joey*", url: "https://joeyturn.github.io/" }
  - { name: "Dhruva*", url: "https://dkarkada.xyz" }
  - { name: "Yuxi", url: "https://yuxi.ml" }
  - { name: "Jamie", url: "https://james-simon.github.io/" }
series: "HEA"
seriesOrder: 1
katexMacros:
  "\\valpha": "{\\bm{\\alpha}}"
---

As a small preview, we can predict *exactly* what the final test error will be if you train a kernel machine on image datasets:

<p style="text-align:center">
  <img src="/hea/learning_curves.png" alt="Kernel curves" width="900" loading="lazy" decoding="async" />
</p>

## Introduction and Motivation

One of the biggest goals of deep learning theory is to explain the underlying black-box machine learning system. One notable observation is that we want to describe how the models react to certain data being presented: we want to be able to understand that a machine sees a cat by the shape of its ears or the color of its fur rather than sporatic correlations between the cat images we gave it during training; we hope that large language models can be understood by presenting it certain sentences to see which parts of the network are activated; we imagine the attention mechanism is putting together an understandable relation between the words we give it; and many, many more examples. In the simplest machine learning system, we've actually already done this: linear regression can be viewed as tending to "learn" (fit the slope to) the parts of the data that have the highest covariance first before fitting those with lower covariance later. While there are many trying to perform a similar in spirit *data-dependent* analysis of large language models, these systems are overwhelmingly complex, enough so to where it isn't clear that tackling the hardest-case, most cutting-edge system is actually the correct way forward. Instead, it is my belief (and once those of the [phage group](https://en.wikipedia.org/wiki/Phage_group) in biology) that we should go to an Occum's razor of models: one that is simple enough to capture key phenomena, and no more complex. 

[Kernel Ridge Regression](https://en.wikipedia.org/wiki/Kernel_method) (KRR) seemed to be a promising avenue--it's a machine learning method that amounts to doing linear regression on the data... just after it's been processed. If you're familiar with kernel regression in detail, feel free to skip to [here](#skipoverview). In math, we can directly compare linear and kernel regression:

$$
\begin{align*}
  f_{\text{linear}}(\vec{x}) &= \sum_i^d m_i x_i\\
  f_{\text{kernel}}(\vec{x}) &= \sum_i^N m_i K(\vec{x}, \vec{x}_i)
\end{align*}
$$

where the main difference to highlight is going from just $\vec{x}$ to $K(\vec{x}, \vec{x}_i)$, the kernel function. Our precursory understanding of KRR comes from applying linear regression on top of the kernel, which relates how similar any two points are $\vec{x}$ and $\vec{x}_i$. Said another way, the kernel is just extracting the distance between any two points, with this distance being in some high-dimensional, nonlinear space, referred to as the 'feature space'.

If you're not yet convinced kernels are the right objects to look at (ie what do these things have to do with practical networks?), you're right to be sceptical! One of the main drivers behind looking at this simple system and hoping our findings will be useful moving forward is any network's relation to the [Neural Tangent Kernel](https://en.wikipedia.org/wiki/Neural_tangent_kernel) (NTK), which are an equivalent description of any network if the width of it is taken to infinity, along with some other conditions on the network. While these NTKs can be arbitrarily complex (specifically, those of the transformer and CNNs), the MLP's neural tangent kernel is rotationally-invariant: in the feature space, distances are calculated solely based off of their difference in angle from the origin. For this reason, we decided to study what the features are of any general rotation-invariant kernel, with a sample scematic of our approach below:

<p style="text-align:center">
  <img src="/hea/data_to_estimator.png" alt="Data transforms to features giving an estimator" width="900" loading="lazy" decoding="async" />
</p>
<!-- ![data_to_estimator](/hea/data_to_estimator.png) -->

<!-- Previous literature on KRR gives not just *exact* analytical expressions for what the training/test error will be as you increase the number of samples you have, but it also tells you exactly what components of the data--the eigenfunctions--contribute most to this error \citep{simon:2021-eigenlearning}. Great! So why not move on and look at MLPs directly then? -->
<!-- Previous works relied on having omniscient knowledge of the feature space that the kernels were using, so actually taking raw data and directly predicting the train/test errors was entirely an empirical endeavor. Our new paper, "Predicting kernel regression learning curves from only raw data statistics" is the last step in completing our understanding in KRR, under some light assumptions. -->

## Setup

Rotation invariant kernels can be written mathematically as

$$
\begin{equation*}
    K(\vec{x}, \vec{x}') = K(\vec{x}^\top \vec{x}')
\end{equation*}
$$

where I've abused notation a bit. For the data itself $\vec{x}$, kernel theory says we need to focus on having some fixed distribution $\mu$ from which we get the data from, so we won't be solving rotation-invariant kernels in totality. Instead, we can note that taking the PCA of a group of images, gives PCA components (which look like blurry images) with marginals being roughly a Gaussian distribution. That means that after taking the PCA of images, any one image can be reconstructed by doing some weighted average of the PCA components, with the weighting being proportional to principal component's singular value! For theoretical understanding, we thus limit our scope of the problem to solving rotation-invariant kernels on anisotropic Gaussian data:

$$
\begin{align*}
    \vec{x} &\sim \mathcal{N}(0, \Gamma)\\
    \Gamma &= \text{diag}(\gamma_0, \gamma_1, \cdots, \gamma_{d-1}).
\end{align*}
$$

## Component-wise understanding of the kernel

As previously alluded to, there are essentially two steps in kernel regression: the kernel is created, then linear regression is performed. Since linear regression is solved[^solved], we just need to understand the kernel in a way that can be used with the linear regression second step. To do this, previous work \citep{simon:2021-eigenlearning} (other references to be added) have found that an eigendecomposition of the kernel is needed: obtaining the eigenvalues (scales) and eigenvectors (directions) of the kernel is enough to directly predict *exactly* what the kernel will do, at least in the average case. 

$$
\begin{align*}
    K(x, x') &= \sum_i \lambda_i \phi_i(x) \phi_i(x')\\
    f(x) &= \sum_i v_i \phi_i(x)
\end{align*}
$$

where I've suggestively wrote the form of our predictor $f(x)$ to resemble that of linear regression, giving some sense for the fact that once the eigensystem is found, we just need to apply linear regression thereafter on the features (eigenfunctions) of the data $\phi_i(x)$. In the previous work, the eigendecompositon was just solved for numerically without any theoretical backing, so the basis of our paper is to do this analytically!

To give a hint as to how to do this, we can write out the eigensystem equation as follows:

$$
\begin{equation}
    \int_{x'\sim\mu} \text{d}x' K(x, x') \phi(x') = \int_{-\infty}^\infty \text{d}x' p(x') K(x, x') \phi(x') = \lambda \phi(x)
\end{equation}
$$

where $\mu$ is the data distribution I mentioned earlier, and $p$ is it's probability distribution. Hopefully as written, it's more clear that the data will determine the *eigen*system of a kernel: any complete basis can be used to describe a kernel, but there is only one that is *eigen* to the data its being evaluated on.

<div id='skipoverview'>

## Kernel-less eigensystem

There's a neat question one can ask about anisotropic Gaussian data on kernels: for a general kernel, what should we expect the eigenfunctions to look like? How much do we even need to worry about the specifics of any kernel? The answer is, surprisingly, not much... let's go through it. You can broadly say, "well, the kernel should be able to represent any polynomial of the data," so we can separate out the features into different levels representing the orders of the polynomials. I'll give the first one as a freebe as I didn't motivate it all that much:

- Const mode: $\lambda$ = 1, $\phi(x)$ = 1

from there on, we can guess that the data $x_i \sim \mathcal{N}(0, \gamma_i)$ itself (ie a linear function of the data) will be a feature[^gammacon]:


- Linear modes: $\lambda$ = $\gamma_i$, $\phi(x)$ = $x_i$.

From here on, I hope the pattern has become more obvious, with the next level being...

- Quadratic modes: $\lambda$ = $\gamma_i\gamma_j$, $\phi(x)$ = $x_ix_j$
- Cubic modes: $\lambda$ = $\gamma_i\gamma_j\gamma_k$, $\phi(x)$ = $x_ix_jx_k$

## Kernel eigensystem

The above picture obviously misses some things, namely *the kernel itself*. However, our guess is **surprisingly** close to the real kernel eigensystem!

## References


[^solved]: Solved is a word that holds quite a bit of baggage; luckily, I haven't found anyone who said linear regression can't be understood in some principled way.

[^gammacon]: We usually like to have $\gamma_i \text{ < } 1$ such that the constant mode comes first, but nothing really breaks if this is not true.

<!-- [^gammacondition]  -->