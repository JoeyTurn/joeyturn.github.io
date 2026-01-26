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

What quantifies a phase transition in neural networks? An adjustable parameter that makes a dramatic change in performance when properly tuned (ie not being able to ‘zoom into’ an x-axis and find a smooth transition)? These sharp drops in performance seem honestly a bit artificial: if there’s a distinct, sharp drop in loss, then that should be the first mode learned by a network, so the fact that some of these drops are learned later on just means the measurement (test error) is just looking at a later mode… why is this unexpected? Is this really a phase transition? The system output should only change marginally…