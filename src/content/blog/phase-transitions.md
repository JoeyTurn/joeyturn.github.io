---
title: Do phase transitions exist in neural networks
pubDate: "2026-01-06"
description: "No"
frontpagedescription: "Phase transitions"
series: "negative"
seriesOrder: 2
draft: True
onlylink: True
---

# Phase transitions in Machine Learning

As we continue to scale our models, it appears that our models have had not just continual performance increases, but that there are certain tasks earlier models could at best guess randomly that are no being performed to extremely high accuracy. The current zeitgeist has painted phase transitions as the primary explanation: that smaller models are <i>qualitatively</i> different than their larger counterparts. As a physicist who found phase transitions fascinating, I was excited to hear contemporary researchers in ML use the term to describe the pheonmenon at large.

As a broad term, there isn't anything wrong with this definition; after all, large networks are indeed performing computations and tasks that smaller networks are seemingly incapable of. I believe, however, that there is a problem with taking the term of "phase transitions" too literally: there is scant evidence that there exist many jumps in performance due directly to having some sort of discontinuous behavior (which is how physicists define phase transitions).

In this post, I will recount some examples of true discontinuities; go over examples of things that appear phase-transition like, but fundamentally seem to [not be]; and finally, go over a case study I have encountered where phase-transition behavior appeared without there actually being any truely distinct phases.

# Phase transitions & Physics
What quantifies a phase transition in neural networks?


An adjustable parameter that makes a dramatic change in performance when properly tuned (ie not being able to ‘zoom into’ an x-axis and find a smooth transition)? These sharp drops in performance seem honestly a bit artificial: if there’s a distinct, sharp drop in loss, then that should be the first mode learned by a network, so the fact that some of these drops are learned later on just means the measurement (test error) is just looking at a later mode… why is this unexpected? Is this really a phase transition? The system output should only change marginally…