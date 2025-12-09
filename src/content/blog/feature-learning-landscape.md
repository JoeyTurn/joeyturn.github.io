---
title: Feature learning networks are/become balanced
pubDate: "2025-10-18"
description: "Feature learning made simple"
series: "Feature Learning"
seriesOrder: 1
draft: False
onlylink: False
---

I've been going through the literature on feature-learning linear networks, with the hope that it be useful to non-linear networks. Along the way, I came across "[From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks](https://arxiv.org/pdf/2409.14623)"\citep{dominé2025lazyrichexactlearning}, and while reading through, it seemed like they had identified a different type of feature learning than that of the maximum-update parameterization ($\mu$P) feature learning that I knew.

For reference $\mu$P \citep{yang:2021-tensor-programs-IV} tells you how you need to scale all of your network's (scale/learning-rate) hyperparameters in order for networks to stabily learn non-trivial functions even at infinite width, with this new perspective based on the ideas of network layers being 'balanced' (drawing back all the way to \citep{NEURIPS2018_fe131d7f}), with \citep{kunin2024richquickexactsolutions} describing how different amounts of balancedness affect feature learning. Note the balancedness picture's lacks of mentioning stability for large width networks: as long as you make two subsequent layers have 'a similar scale,' that's all you need!

In this post, I want to discuss how the ideas of $\mu$P align with those of deep linear network balancedness in a unified way, and use this perspective to make predictions for how other 'ML trickery' that is often used in practice promotes feature learning.

### Setup

Unless otherwise stated, all networks will be 2 layer (1 hidden layer) networks of the form $\mathbf{f}(\mathbf{x}) = \sum_{i} \mathbf{a}_i \sigma(\sum_jW_{ij}x_j)$. The input dimension of $d$ ($x \in \mathbb{R}^{d}$) and output dimension of $d_{out}$ ($\mathbf{f}(\mathbf{x})\,, \mathbf{y} \in \mathbb{R}^{d_{out}}$). The nonlinearity is to be taken as the identity function $\sigma: \mathbf{x}\mapsto \mathbf{x}$. The hidden dimension $w$, $W \in \mathbb{R}^{w \times d}$ is a parameter we can scale, where we wish for the network to be trainable for any value of $w$. The network is to be updated iteratively using gradient flow (without addons such as weight decay/momentum/etc.) $\Delta \mathbf{a} = -\eta_a \frac{\partial \mathcal{L}}{\partial \mathbf{a}}$, $\Delta W = -\eta_W \frac{\partial \mathcal{L}}{\partial W}$, ($\eta_a, \eta_W \ll 1$) under a square loss $\mathcal{L} = (\mathbf{f}(\mathbf{x})-\mathbf{y})^2$.[^bsz]

Occasionally, I will denote $X \in \mathcal{R}^{N\times d}$ and $Y \in \mathcal{R}^{N\times d_{out}}$ later during discussion. These are matrices with rows consisting of individual samples: $X_i/Y_i = \mathbf{x}^{(i)}/\mathbf{y}^{(i)}$.

## How to achieve feature learning

### $\mu$P

Based on \citep{yang:2023-spectral-scaling}, we should set up our network with the following conditions:

$$
\begin{align*}
    \sigma_W &= \Theta\left(\sqrt{\frac{1}{d}}\right)\\
    \sigma_a &= \Theta\left(\sqrt{\frac{d_{out}}{w^2}}\right)\\
    \eta_W &= \Theta\left(\frac{w}{d}\right)\\
    \eta_a &= \Theta\left(\frac{d_{out}}{w}\right)
\end{align*}
$$
where $\mathbf{a}_i \sim \mathcal{N}(0, \sigma_a^2)$, $W_{i,j} \sim \mathcal {N}(0, \sigma_W^2)$. The broad intuition for this setup is such that we have (1) non-trivial updates to our weights (more specifically, the first weight matrix W), else we end up with a NNGP, and (2) not have our outputs blow up during training. Up to moving around the factors (à la mean-field parameterization), this has been identified as the only ($d$, $w$, $d_{out}$)-unique scaling of our network parameters.

### Balancedness

If that is true, then why does \citep{dominé2025lazyrichexactlearning} have a seemingly different rule for achieving feature learning that is removed from the $\mu$P scaling? Their paper finds feature learning for a (deep) linear network can be achieved through the following condition,[^balancednessreqs] modified slightly to accomidate layerwise learning rates:

<div id="eq1">

$$
\begin{equation}
    \text{At init, } \frac{1}{\eta_a}\mathbf{a}^\top \mathbf{a} - \frac{1}{\eta_W} WW^\top = \lambda I_w,
\end{equation}
$$
</div>

with data being whitened: $$X^\top X = I_d$$.

A proof for this quantity remaining invariant can be found in both \citep{kunin2024richquickexactsolutions} and [below](#conserved). With Gaussian initialization, this conserved equation is trivially achieved when the width w is large, making

$$
\begin{equation*}
    % \mathbf{a}^\top\mathbf{a} &\sim d_{out}\sigma_a^2 I_w\\
    % WW^\top &\sim d\sigma_W^2 I_w\\
    \frac{1}{\eta_a}\mathbf{a}^\top\mathbf{a}-\frac{1}{\eta_W}WW^\top\sim\left(\frac{1}{\eta_a} d_{out}\sigma_a^2-\frac{1}{\eta_W} d\sigma_W^2\right)I_w \equiv \lambda I_w,
\end{equation*}
$$

where $\sim$ is used due to sub-leading order corrections not being included; from here on, I will be using $=$, with the understanding that this is to leading order. Thus far, we haven't needed to worry about making sure update sizes won't cause trivial or blown-up updates at all due to the trajectory of our network lying on an invariant manifold defined by [Eq. 1](#eq1), so what gives? The trick is in the value that $\lambda$ takes: the closer to 0, the more our solutions appear to be in the feature-learning regime.

While our solution space for feature learning is quite large, with 3 different independent knobs to turn, if we rewrite our feature learning condition in the following way,

$$
\begin{equation*}
    \lambda = \frac{1}{\eta_a} d_{out}\sigma_a^2 - \frac{1}{\eta_W} d\sigma_W^2 = \frac{d_{out}}{\eta_a}\sigma_a^2 - \frac{d}{\eta_W}\sigma_W^2 \leq \epsilon
\end{equation*}
$$

for some small $\epsilon$ that'll represent how close to the "true" feature learning network we get, we can see a 'natural choice' for how we should scale our hyperparameters; taking both portions of the equation to be invariant to our network parameters of $(d, d_{out}, w)$ forces $\sigma_a^2 \sim \eta_a/d_{out}$, $\sigma_W^2 \sim \eta_W/d$, representing a perfect match with how \citep{yang:2023-spectral-scaling} told us to scale, up to actually choosing the layerwise learning rates and reintroducing a factor of $w$ (which I interpret as a byproduct of moving from a $\lambda$-rescaled diagonal $w\times w$ identity matrix to a single scalar $\lambda$).
<!-- which I believe has to do with the Gram matrix size being w x w, Jamie/Dhruva correct me if wrong. -->

For completeness, let's plug in the $\mu$P scaling laws to see how small $\lambda$ is:

$$
\begin{equation*}
    \lambda = \Theta\left(\frac{w}{d_{out}}d_{out}\frac{d_{out}}{w^2}-\frac{d}{w}d\frac{1}{d}\right)=\frac{f_2d_{out}}{w}-\frac{f_1d}{w}=\frac{f_2d_{out}-f_1d}{w}
\end{equation*}
$$

where $f_2$ and $f_1$ represent scalars that are $\Theta(1)$ with respect to the width. This quantity tends towards 0 as $w\rightarrow \infty$ and tells us that any scaling done to $d$ XOR $d_{out}$ should be reflected in an equivalent change to $w$: $w \sim d(_{out})$.

### Lazy & ultra-rich scaling

On first glance, we have more freedom in the balancedness picture than the $\mu$P picture, so what gives? Lazy networks have thus far been more approachable from a theoretical perspective, so surely balancedness should be our preferred lens with which to view feature learning. Why are people so much more drawn to $\mu$P, outside of it being the first to introduce many of the feature learning concepts we know today?

Taking from the MFP literature, if we scale our network outputs as $ f \mapsto f/\gamma$ and the overall learning rate as $\eta \mapsto \eta\cdot\gamma^2$ for $\gamma \leq 1$ and $\eta \mapsto \eta\cdot\gamma$ otherwise, a scaling law discovered in \citep{atanasov:2024-ultrarich}, we can transition a network into the lazy regime with $\gamma \ll 1$ and into a new 'hyper-rich' regime of $\gamma \gg 1$. In the balancedness picture, both the network output prefactor and learning rate rescaling can be absorbed globally; it's unclear where to put the network output prefactor, but the learning rate can be easily included: 

$$
\lambda \sim
\begin{cases}
  \left(\frac{1}{\eta_a} d_{out}\sigma_a^2-\frac{1}{\eta_W} d\sigma_W^2\right)/\gamma^2, & \gamma \leq 1\\
  \left(\frac{1}{\eta_a} d_{out}\sigma_a^2-\frac{1}{\eta_W} d\sigma_W^2\right)/\gamma, & \gamma \text{ > } 1.
\end{cases}
$$

plugging in our $\mu$P relations, this gets

$$
\lambda(\omega) = \Theta\left(\frac{d_{out}-d}{\omega}\right),\\ 
\omega \equiv 
\begin{cases}
    w\gamma^2, & \gamma \leq 1\\
    w\gamma, & \gamma \text { > } 1.
\end{cases}
$$

Some insight that comes from this setup is that lazy networks should scale their richness paramater $\gamma$ such that $\gamma \gg \sqrt{w}$, or else we run the risk of not leaving the rich regime. No such insight appears to be able to be gleamed from the ultra-rich case, where both the width and the richness parameter contribute to lowering $\lambda$. Finally, our equation recovers $\lambda \rightarrow 0 (\pm\infty)$ as $\gamma \rightarrow \infty(0)$, as expected. 

This suggests that $\mu$P is not the only way to see feature-learning networks; we an also look at how balanced a network is throughout training in order to gain a deeper understanding of its feature learning properties. 

## Big picture: why (insert ML trick) lets you learn features

Up until now, we've considered an extremely simple network with no special bells or whistles. If we want to understand how different ML levers affect feature learning, we need to move beyond this, and now that we have the balancedness-$\mu$P picture, we have a greater capacity to add on extra parts.

### Weight decay

Take (L2) weight decay, for example: the purpose is to decrease the weights of a network. Previous papers [links] have investigated ...

There doesn't exist an invariant when training is done with weight decay (except under the explicit case that the layerwise learning rates are identical and additionally the weight decay constant is global), but we can see how our earlier constant $\lambda$ varies with the weight penalty hyperparameter $\zeta$:

$$
\begin{equation*}
    \frac{\text{d}\lambda}{\text{d}t} = -2\zeta\left(\lvert\lvert \mathbf{a}\rvert\rvert_F-\lvert\lvert W\rvert\rvert_F\right)
\end{equation*}
$$

where we note that weight decay lowers the overall scale of weights by forcing the weights between layers to become balanced, forcing the network parameters onto a 'more feature-learning manifold'.

### Learning rate warmup

To analyze learning rate increases (and subsequent decays), we need to move away from gradient flow. We'll still let $\lambda = \lvert\lvert a\rvert\rvert_F^2/\eta_a-\lvert\lvert W\rvert\rvert_F^2/\eta_W$ and now have $\Delta\lambda = \eta_a\lvert\lvert R(WX)^\top\rvert\rvert_F^2-\eta_W\lvert\lvert \mathbf{a}^\top R X^\top\rvert\rvert_F^2$ for residual $R \equiv Y-aWX$. If we define $M = XR^\top$, we can rewrite $\Delta \lambda$ as,

$$
\begin{align*}
    \Delta \lambda(t) &= \eta_a(t) \alpha_t - \eta_W(t)\beta_t \leq -\eta_a\eta_W \lvert\lvert M\rvert\rvert_F^2 \lambda\\
    \alpha_t &= \lvert\lvert WM\rvert\rvert_F^2\\
    \beta_t &= \lvert\lvert a^\top M\rvert\rvert_F^2
\end{align*}
$$

with the general interpretation being for any initial values $\lambda$ far from 0, learning up the learning rate allows the weights to take much faster progress towards $\lambda=0$; this needs to be followed up by a decrease in the learning rate, or else $\lambda$ will end up "bouncing" between the positive and negative of some final value $\pm \lambda_f$.


### Deeper network balancedness

If we want to consider deeper networks, the story is largely the same. Take $Y=W_LW_{L-1}\cdots W_2W_1X$. Between two subsequent layers, we have

$$
\begin{equation*}
\frac{1}{\eta_{\ell+1}}W_{\ell+1}^\top W_{\ell+1}-\frac{1}{\eta_\ell}W_{\ell} W_{\ell}^\top = \lambda_\ell I_{w_\ell}
\end{equation*}
$$ 

that is, $W_{\ell+1}^\top W_{\ell+1}$ and $W_{\ell} W_{\ell}^\top$ are simultaneously diagonalizable, giving

$$
\begin{equation*}
\sigma_i(W_{\ell+1})^2= \sigma_i(W_{\ell})^2+\lambda_\ell
\end{equation*}
$$

while we can't generally compare $\frac{1}{\eta_i} W_{i}^\top W_{i} - \frac{1}{\eta_j} W_{j} W_{j}^\top$ ($j \neq i-1$) due to possible shape inconsistency, we can say

$$
\begin{equation*}
\sigma_i(W_{\ell+1})^2= \sigma_i(W_{k})^2+\sum_{i=k}^{\ell-1}\lambda_i
\end{equation*}
$$

with the implication being that there is a general balancedness between all layers that feature learning networks satisfy: the $i$-th singular values of all layers in a feature learning network are all roughly equivalent.

## Takeaway & Future Efforts

This blogpost was written as a dive into something I found that didn't make sense: how can balanced nets fit into the $$mu$$P picture. While discussing my thoughts with Dan Kunin (whose work is the core of this post), we noted how I'm using the known $$\mu$$P to get balacedness. Right now, I'm working on extending this in the opposite direction: can we get a different perspective on feature learning through balacedness <i>alone</i>, no $$\mu$$P involved? This is (part of) my current research, so if that sounds interesting, please feel free to reach out!

## Appendix

<div id='conserved'>

### A1: Conserved balancedness throughout training 

*To be written; check Dan Kunin's 

</div>

## References


[^bsz]: It should be mentioned that in my experiments, I use a batch size instead of population gradient descent, although I suspect this minimally changes results.

[^balancednessreqs]: The paper itself details one other requirements: non-bottleneckedness/non-overparameterized. While the first condition can be trivially dropped for our discussion, we always assume $w$ > $\max(d, d_{out})$; from what I can tell, the non-overfitting condition can be dropped for our discussion.