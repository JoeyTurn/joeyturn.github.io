---
title: Kernels are data augmentation-invariant
pubDate: "2026-02-03"
description: "Insights into data augmentation in the kernel and feature-learning regimes"
series: "Feature Learning"
seriesOrder: 1
draft: False
onlylink: False
---

# Fundamental Understanding of Data Augmentation

I would argue that there are not any results that detail <i>when</i> data augmentations are helpful, and by <i>how much</i> they will improve the performance. As kernel theory has nice theory=experiment curves, I tested out running three of the most common data augmentation techniques (rotations, random cropping, and horizontal flips) on CIFAR10 through a ReLU NTK and Convolutional NTK.

To give the punchline early, I largely found that data augmentation is a feature-learning regime specific phenomenon!

<figure style="margin: 0;">
    <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    <img src="/data_augments/da35mlphflip.png" alt="Feature-learning regime MLP"
        style="width:100%; max-width:450px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <img src="/data_augments/da35lazymlp.png" alt="Kernel regime MLP"
        style="width:100%; max-width:450px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    </div>
    <figcaption style="text-align:center; font-size:0.95em; color:#666; margin-top:8px;">
        <b>Left.</b> Trained on a feature-learning version of an MLP, adding horizontal flips as data augments increases performance on the test set. <b>Right.</b> Training an MLP in the lazy/kernel regime, the number of horizontal flipped examples has seemingly no correlation with the performance.
  </figcaption>
</figure>

## Kernel findings

What is the optimal number of data augmentation samples to produce for a given model and dataset? It's a simple question that I couldn't find the answer to, even in the most simplistic models! Given how the most highly powered models are quickly running out of raw training data, I find the question ever more pertinent.

To start with my take, I wanted to look at kernel ridge regression. After a visit to NeurIPS, my initial guess was that performance would cap out around $log(|G|)$ if the size of our data augmentation group (when applicable) contained $|G|$ total elements, based on a precursory understanding of the [rank of groups](https://en.wikipedia.org/wiki/Rank_of_a_group).
For the kernel of choice, I chose the ReLU NTK as I found it to be a highly powerful, yet [tractable](http://arxiv.org/abs/2510.14878) function. 

I first trained on airplane (0) vs frog (6), which is to my knowledge the easiest task for CIFAR10. All datapoints presented in this blogpost are taken over 3 random initializations.

<figure style="margin: 0;">
    <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    <img src="/data_augments/da06relurot_large.png" alt="ReLUNTK with Rotations"
        style="width:100%; max-width:450px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <img src="/data_augments/da06relurot.png" alt="Full ReLUNTK with Rotations"
        style="width:100%; max-width:450px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    </div>
    <figcaption style="text-align:center; font-size:0.95em; color:#666; margin-top:8px;">
        Each dot represents a trained kernel solution with low ridge (1e-3). Black represents kernels trained without any augments. Each colored curve represents one fixed number of base unaugmented data, data augments being added as the number of training samples increase. <b>Left.</b> Training a ReLU NTK on CIFAR10 0 vs 6 with rotations yields no performance differences based on amount of rotations. <b>Right.</b> Zoomed out version of left, including low n performance.
  </figcaption>
</figure>

Surprisingly, there is <b>no</b> noticable difference between the kernel performance when applying (rotational) data augments!

There could be a few effects explaining this: the chosen task was too easy, the choice of kernel make rotations not affect performance, or maybe it was a poor choice of data augment. I'll go through each of these individually, but will only be plotting the higher sample runs as there's nothing particularly interesting with the low sample ones.

### Task change

Was the task too easy? In terms of kernel theory, there's not much credible evidence that changing the task would have much of an impact to our approach: the only difference we would see is if the eigenfunction corresponding with some general notion of our augment was both low in eigenvalue and high in eigencoefficient, which is highly unlikely, yet not impossible. To double check, I changed the task to predict cat (3) vs dog (5), and stuck with this throughout the remaining studies.

<figure style="margin: 0;">
    <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    <img src="/data_augments/da35relurot.png" alt="ReLUNTK on CatDog with Rotations"
        style="width:58%; max-width:900px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <figcaption style="text-align:center; font-size:0.95em; color:#666; margin-top:8px;">
        Changing task distribution appears to have no effect on our findings.
  </figcaption>
</figure>

As predicted, changing our task doesn't change our results. Neat for rotations aren't low in eigenvalue and high in eigencoefficient, but not much beyond that.

### Kernel change

Depending on how much you know about kernels, an argument can be raised that the ReLU NTK shouldn't change when adding in rotational data augments as the kernel itself is rotation-invariant. Although the rotation-invariance of the kernel is in the sample-space as opposed to the pixel-space that the augmentations themselves are happening in, I wasn't entirely sure if this would change anything, albeit I had my doubts. To test, I used the simplest Convolutional NTK I could come up with (average patch embeddings $\rightarrow$ 2 layer ReLU NTK); perhaps this was too simple? Regardless, the rotation-invariance was broken, and rotational data augments still had no effect on the kernel:

<figure style="margin: 0;">
    <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    <img src="/data_augments/da35cntkrot.png" alt="ConvNTK on CatDog with Rotations"
        style="width:58%; max-width:900px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <figcaption style="text-align:center; font-size:0.95em; color:#666; margin-top:8px;">
        Changing the kernel model appears to have no effect on our findings.
  </figcaption>
</figure>

### Data augment change

It could be that the rotational data augment is just extremely underpowered and barely changes performance. To test if this was true, I asked ChatGPT for the most impactful augmentations (mostly because I'm unfamiliar with what actually helps) and it came up with horizontal flips and random cropping. Both seemed to have a few results on kernels that boosted performance (when the full CIFAR dataset was used), but it doesn't seem like anyone had tested the effects in general!

<figure style="margin: 0;">
    <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    <img src="/data_augments/da35cntkcrop.png" alt="ConvNTK with Cropping"
        style="width:100%; max-width:450px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <img src="/data_augments/da35cntkhflip.png" alt="ConvNTK with Horizontal Flips"
        style="width:100%; max-width:450px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    </div>
    <figcaption style="text-align:center; font-size:0.95em; color:#666; margin-top:8px;">
        Yet again, the choice of data augment seemingly has no effect on performance.
  </figcaption>
</figure>

I could not find a case where data augments improved performance on kernel models! I'm surprised no one has done these tests, at least not to my knowledge!

At this point, I had my suspicions that data augments were entirely useful only in the feature-learning regime; after all, they still <i>are</i> used and not completely without merit, as far as I knew.

## MLP findings

I used [modelscape](https://joeyturn.github.io/blog/modelscape/) to train MLPs in the feature-learning regime ($\mu$P) with the same general setup. A few tweaks had to be made, however: to ensure I got results eventually, I set the trainloop had a hard limit of $10^4$ gradient steps being taken before exiting.[^exitcond] Running off of the same three data augmentation techniques as before, we finally see the benefits of adding synthetic samples: 

<figure style="margin: 0;">
    <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    <img src="/data_augments/da35mlprot.png" alt="FL MLP with Rotations"
        style="width:100%; max-width:350px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <img src="/data_augments/da35mlpcrop.png" alt="FL MLP with Cropping"
        style="width:100%; max-width:350px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <img src="/data_augments/da35mlphflip.png" alt="FL MLP with Horizontal Flips"
        style="width:100%; max-width:350px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    </div>
    <figcaption style="text-align:center; font-size:0.95em; color:#666; margin-top:8px;">
        Feature-learning networks have improved performance when trained with data augmentation.
  </figcaption>
</figure>

Great! Data augmentations are actually useful, as we expected. The story here then seems to be that data augmentation is useless on kernel methods altogether...

### Kernel MLP training

To test this, I lowered the richness parameter $\gamma$ to $10^{-3}$ (more details about the richness parameter discussed in \citep{atanasov:2024-ultrarich}) to check that a kernelized MLP (ie one that is well descibed by a second order Taylor series) indeed is data-augment agnostic. For the sake of time, I only tested on horizontal flips, although I strongly believe the findings will hold for the other choices.

<figure style="margin: 0;">
    <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    <img src="/data_augments/da35lazymlp.png" alt="Kernelized MLP with Horizontal Flips"
        style="width:100%; max-width:450px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <img src="/data_augments/da35mlphflip.png" alt="FL MLP with Horizontal Flips"
        style="width:100%; max-width:450px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    </div>
    <figcaption style="text-align:center; font-size:0.95em; color:#666; margin-top:8px;">
        Kernelized MLPs don't get performance gains from data augmentation.
  </figcaption>
</figure>

Unsurprisingly, data augments <i>only</i> help the networks I've trained in the feature-learning regime. I largely take this as a sign that a fundamental study of what data augments do to a network--when they help, how many to include, how much the performance gain will be--will come only from studying feature-learning networks, and that a kernel picture of data augmentation is simply impossible, least there be some specific augment tailor-made for kernels.

## References

[^exitcond]: There was also an exit condition once a test error of 0.1 was achieved (the labels have are $\pm 1$, so the model must train to get to this threshold), though for the number of samples I was using, this almost never happened.