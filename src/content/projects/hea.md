---
title: Rundown of "Predicting Kernel Regression Learning Curves from only Raw Data Statistics" - the HEA
pubDate: "2025-10-11"
description: "On real data, we know exactly how kernel regression performs."
links:
  - { label: "arXiv",  url: "https://arxiv.org/abs/1234.5678" }
  - { label: "GitHub", url: "https://github.com/james-simon/feature_recombination" }
authors:
  - { name: "Joey*", url: "https://joeyturn.github.io/" }
  - { name: "Dhruva*", url: "https://dkarkada.xyz" }
  - { name: "Yuxi", url: "https://yuxi.ml" }
  - { name: "Jamie", url: "https://james-simon.github.io/" }
series: "HEA"
seriesOrder: 1
---

As an ML theorist (in training), one of the grand goals of the field is commonplace in most burgeoning theoretical disciplines: *how can we help guide experimentalists design better systems*? We've broadly realized that the most cutting-edge systems, take OpenAI's ChatGPT, are far more complex than what we can currently analyze, so we simplify our systems and gradually build up. 

[Kernel Ridge Regression](https://en.wikipedia.org/wiki/Kernel_method) (KRR)--a machine learning method amounting to doing linear regression of data in a transformed space--gained a lot of traction within the past few years on account of the [Neural Tangent Kernel](https://en.wikipedia.org/wiki/Neural_tangent_kernel) (NTK), a kernel equivalent to neural networks (with a certain setup). The upside? KRR is a lot more simple to analyze than the networks they're traditionally associated with: Multi-Layer Perceptrons (MLPs). Previous literature on KRR gives not just *exact* analytical expressions for what the training/test error will be as you increase the number of samples you have, but it also tells you exactly what components of the data contribute most to this error [citation].