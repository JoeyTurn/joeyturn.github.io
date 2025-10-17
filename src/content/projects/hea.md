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

In ML theory, one of the grand goals of the field is to actually be useful to experimentalists. We've broadly realized that the most cutting-edge systems, take OpenAI's ChatGPT, are far more complex than what we can currently analyze, so we simplify our systems and gradually build up. 

[Kernel Ridge Regression](https://en.wikipedia.org/wiki/Kernel_method) (KRR)--a machine learning method amounting to doing linear regression of data in a transformed space (the feature space)--gained a lot of traction within the past few years on account of the [Neural Tangent Kernel](https://en.wikipedia.org/wiki/Neural_tangent_kernel) (NTK), a kernel equivalent to neural networks (under a certain setup). The upside? KRR is a lot more simple to analyze than the networks they're traditionally associated with: Multi-Layer Perceptrons (MLPs). Previous literature on KRR gives not just *exact* analytical expressions for what the training/test error will be as you increase the number of samples you have, but it also tells you exactly what components of the data--the eigenfunctions--contribute most to this error \citep{simon:2021-eigenlearning}. Great! So why not move on and look at MLPs directly then?
Previous works relied on having omniscient knowledge of the feature space that the kernels were using, so actually taking raw data and directly predicting the train/test errors was entirely an empirical endeavor. Our new paper, "Predicting kernel regression learning curves from only raw data statistics" is the last step in completing our understanding in KRR, under some light assumptions.

## Setup

Throughout the paper, we will always consider the kernels to be rotation-invariant. That is, all kernels will be able to take the form of 

$$
\begin{equation}
    K(\valpha) = 
\end{equation}
$$

## Predictions of an eigensystem

Before discussing 

## References