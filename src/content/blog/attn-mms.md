---
title: Linear attention is unable to perform the In-Context Markov Model/In-Context Learning of Representations task
pubDate: "2026-06-24"
frontpagedescription: "Linear attention is not the minimal model to study ICLR"
description: "Linear attention is not the minimal model to study ICLR"
series: "Transformers"
seriesOrder: 1
draft: False
onlylink: False
---

# Why ICLR?

From the original [ICLR](https://arxiv.org/abs/2501.00070)\citep{park2025iclrincontextlearningrepresentations} paper, I joined the other ICL theorists in being enamored with the result.
Language models could reconstruct a random walk process over a random grid, the states of which are stored in context!
This appeared to be qualitatively different from the standard ICL tasks that were considered before, as the model's representations never really needed to be discussed.
Why was this happening? Why did the model construct the generative process within its representations? There were so many things to explore about this setup, and I was looking for some simple insight.

# Minimal models

As a theorist, one of the first steps is to consider what minimal system or model will reproduce all of the aspects that we wish to extract from the larger system without being overly complex.
For the usual ICL tasks, this amounted to restricting the task itself to be in-context regression problems (given $\mathbf{X}$, $\vec{y}\approx\mathbf{X}\vec{\beta}$ pairs in context and asked to compute what the optimal linear weight $\beta$ maps $\mathbf{X}$->$\vec{y}$, ie $X_1=1, y_1=3, X_2=-2, y_2=-6, X_q=4, y_q=?$), or in-context classification problems (given $\mathbf{X}$, $\vec{y}$ where $\vec{y}$ takes a few set values, find the local regions of $X$ that predict y, ie $X_1 = 3, y_1 = a, X_2 = -1, y_2 = b, X_3 = 3.5, y_3 = a, X_4 = 2.8, y_4=?$), with the simplest possible system to analyze these systems appearing[^notactually] to be a single linear attention layer.
This begs a natural line of inquiry: can we use this same one-layer linear attention model to analyze the ICLR setup?
If this linear attention layer is enough, why? Are there nice mathematical simplifications that can be made such as those in the "asymptotic theory of in-context learning by linear attention" \citep{Lu_2025}?
If it isn't enough, why not? And what will be enough?

## ICLR solution

To decide if our system is capable of performing the ICLR task, it will be helpful to know <i>what</i> we are aiming for, what the true predictor should be.
To perform an ICLR-like task, which consists of random walks on a grid performed in-context, our solution needs to perform two different operations: (1) construct the transition probability matrix, which dictates how likely it is to go from one position on our graph to all other positions on the graph, and (2) apply this transition probability matrix as an operator onto our current graph state.
There's possibilites for this to open up to predicting what the graph state will be not just at the next step, but at a T steps in the future, but we'll ignore this for now. 

Linear algebra is great at applying matrices to vectors, so (2) applying the matrix to our current state should be simple for our model to do. The load bearing step is thus finding the transition matrix! How do we do this?
Broadly speaking, finding the transition matrix consists of two steps. Firstly, we need to find the overlap between all states and what the subsequent state is, let's call this $\mathbf{C}$ for a correlation matrix: $\mathbf{C} = \sum_{t=1}^{L-1} \vec{x}_t\vec{x}_{t+1}^\top$ where $\vec{x} \in \mathbb{R}^{d}$ are the states/locations of the graph, which we can imagine are one-hots for now, and $t \in [L]$ represents the location within the context.
Secondly, we need to ensure these are proper probability maps, such that each row of $\mathbf{C}$ sums to 1. We'll do this by normalizing each row of $\mathbf{C}$ using $\mathbf{D} = \sum_{t=1}^{L-1} \vec{x}_t\vec{x}_{t}^\top$, with our final probaility matrix being $\mathbf{T} = \mathbf{D}^{-1}\mathbf{C}$.
Moving forward, I'll change the explicit sums of outer products of vectors to be matrix multiples over $\mathbf{X}$, with $\mathbf{C} = \mathbf{X}^\top\mathbf{Y}$, $\mathbf{D} = \mathbf{X}^\top\mathbf{X}$, and $\mathbf{T} = \left(\mathbf{X}^\top\mathbf{X}\right)^{-1}\mathbf{X}^\top\mathbf{Y}$ where $\mathbf{X} \in \mathbb{R}^{L \times d}$ consists of rows defined by $\{\vec{x}\}_{t=1}^{L-1}$, and $\mathbf{Y} \in \mathbb{R}^{L \times d}$ consists of rows of the same data as $\mathbf{X}$, just shifted one position in the context: $\{\vec{x}\}_{t=2}^{L}$.

If we constrain the problem by ignoring the normalization $\mathbf{D}$, which can be done with precise choices of the data's scale, we're left with something that's fundamentally third order in the data: $\vec{y}_{pred} = \vec{x}_{L}\mathbf{X}^\top\mathbf{Y}$, and with the linear attention model explicitly as something third order, with $Attn(\mathbf{X}) = \left(\mathbf{X}\mathbf{W}_Q\mathbf{W}_K^\top\mathbf{X}^\top\right)\mathbf{X}\mathbf{W}_V$, it appears we just need $\mathbf{W_q}$ to zero out everything besides the last row, and $\mathbf{W}_V$, potentially combined with $\mathbf{W}_K$, to shift the context by one place so we can recover $\mathbf{Y}$.

The trouble here is the last part of our setup: both $\mathbf{W}_V$ and $\mathbf{W}_K$ are $d \times d$ matrices, and thus cannot generally couple information across the context. To see this more easily, consider taking an SVD of the context $\mathbf{X}$ as $\mathbf{X} = \mathbf{U}\mathbf{\Sigma} \mathbf{V}^\top$, where $\mathbf{U}^\top \in \mathbb{R}^{L \times L}$ encodes the intra-context information. Applying this to the linear attention formula, we get 
$$
    Attn(\mathbf{X}) = \left(\mathbf{U}\mathbf{\Sigma} \mathbf{V}^\top \mathbf{W}_Q\mathbf{W}_K^\top\mathbf{V}\mathbf{\Sigma}^\top \mathbf{U}^\top \right)\mathbf{U}\mathbf{\Sigma} \mathbf{V}^\top\mathbf{W}_V = \mathbf{U}\mathbf{\Sigma} \mathbf{V}^\top \mathbf{W}_Q\mathbf{W}_K^\top\mathbf{V}\mathbf{\Sigma}^\top \mathbf{\Sigma} \mathbf{V}^\top\mathbf{W}_V
$$
as $\mathbf{U}^\top \mathbf{U} = \mathbf{I}_L$ by construction. Although long, this equation is insightful for what it lacks: the only location where context information is stored or could be potentially modified is at the very beginning! We're computing $Attn(\mathbf{X}) \sim \mathbf{U}\mathbf{M}$ where $\mathbf{M}$ is some complicated matrix that only will modify the $d$ dimensional space, not the context! As the model cannot represent the shift in position required to get $\mathbf{Y}$, we see that linear attention alone is <i>unable</i> to perform random walks on grids, nor the Markov models that generalize them, in-context![^contextshifting] 

# If not linear attention, then what?

With the unfortunate news that linear attention fails at the ICLR task, then is the simplest model that can perform the task?
To start off with another easy one, infinitely deep linear attention also should fail by the same argument, so long as we're interested in the nominal ICLR solution of finding the transition matrix and applying it; infinitely deep linear attention should be able to solve the ICLR solution, just in a wholey uninteresting way due to the universal approximation theorem.
For softmax attention, things look more promising: the softmax function is computed on the $\left(\mathbf{X}\mathbf{W}_Q\mathbf{W}_K^\top\mathbf{X}^\top\right)$ portion of the attention, which does become a $L \times L$ matrix, and so it should be able to this ICLR task.
My hated and beloved kernels/MLPs should also be able to perform this ICLR task, although it appears a single layer of softmax attention is the most readily amenable to theoretical study. If you want to study ICLR at a model-level, that would be one of my top bets to look at (which is also something I'll do in the coming weeks).

<!-- \mathbf{X} -->
<!-- <figure style="margin: 0;">
    <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    <img src="/data_augments/rotation_data_augments_relu_0_6.png" alt="ReLUNTK on Airplane vs Frog with Rotations"
        style="width:58%; max-width:900px; height:auto; display:block;"
        loading="lazy" decoding="async" />
    <figcaption style="text-align:center; font-size:0.95em; color:#666; margin-top:8px;">
        Black represents kernels trained without any augments. Each colored curve represents one fixed number of base unaugmented data, data augments being added as the number of training samples increase. There is no clear trend with increasing the number of augmented samples.
  </figcaption>
</figure> -->

## References

[^notactually]: As I'm currently working through how kernels perform in-context learning, I'm both very convinced and not convinced that linear attention is the simplest model. I think people just haven't looked at kernels in the right way before.

[^contextshifting]: The obvervant will notice that one could construct a shifting matrix within the $d$-dimensional space, however this will only be useful if our task is one specific random walk; as soon as we introduce another random walk with a slightly different transition matrix, the model would need to do an amalgamation of different $d$-dimensional approximations of the shifting matrix, which <i>should</i> result in the model predicting nonsense. I quickly did some math on a spare sheet of paper and figured it wasn't worthwhile as the general approach of constructing the shifting matrix doesn't work, but it's unclear to me how this changes if there are only a few different potential Markov models/random walks considered.