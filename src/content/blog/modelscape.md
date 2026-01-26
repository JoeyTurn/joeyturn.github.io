---
title: Machine learning mega training experiment repo released!
pubDate: "2026-01-26"
description: "Training models now made much simpler!"
frontpagedescription: I made a repo that makes training *any* ML model simple!
series: "Repo Release"
seriesOrder: 2
draft: False
onlylink: False
links:
  - { label: "Link to Repo on GitHub", url: "https://github.com/JoeyTurn/modelscape" }
---

# Introduction + Using

I made [**modelscape**](https://github.com/JoeyTurn/modelscape), a repo extending [**MLPscape**](https://github.com/JoeyTurn/MLPscape). This repo allows for quick iteration of ML experiments without the hassle of needing to change the trainloop, a massive list of for loops, worrying about .py vs .ipynb differences, multiprocessing, and so on!

In this post, I'll detail the updates I've made to create `modelscape`, then will describe what got carried over from `MLPscape` in case you've already used it.

# New in Modelscape 


# Previous usage carried over from MLPscape

If you've used & understood how to use MLPscape, feel free to skip this section! Otherwise, this is a rundown of the main functionality of `modelscape`.

To use *modelscape*, simply follow one of the provided notebooks/python files in the `examples` folder! Typically, your file will look something roughly along the lines of

- Imports
- Your model grabs definitions
- Hyperparameter specification
- Iterator specification
- Data selection
- Batch function selection
- Other pre-trainloop setup
- Trainloop execution
- Results

where most of the code powering *MLPscape* is hidden away in what I call the backend, which handles multiprocessing, the trainloop which runs any functions throughout, and so on. Let's run through each step, just as a reference to come back to in case anything goes wrong:

## Imports

This is quite standard, with the only things that really need attention being importing the trainloop, and the arguments creators (parse_args for .py or base_args for .ipynb; as well as build_other_grabs):

```python
import numpy as np
import torch

import torch.multiprocessing as mp
from modelscape.backend.cli import parse_args, base_args, build_other_grabs
from modelscape.backend.job_iterator import main as run_job_iterator
from modelscape.backend.utils import ensure_torch
```

## Your MLP grabs definitions

Want to check something out while your model is training? Great! Just define whatever function here, and we'll worry about it later. 

The important parts here is to always take **kwargs, and use `model`, `X_tr`, `y_tr`, `X_te`, `y_te`, or other variables that are being iterated over! So far, this is about the most general way I've thought of having the trainloop not need to be changed.

```python
def my_mlp_grab(model, X_tr, y_tr, **kwargs):
    return (y_tr.T @ model(X_tr)).squeeze()
```

## Hyperparameter selection

We'll start here by taking in the default args, and changing whatever we don't want! A list of the default arguments can be found at the bottom of this post, and in the README of the repo.


```python
args = parse_args() OR base_args()

args.ONLINE = False
args.N_TRAIN=4000
args.N_TEST=1000
args.N_TOT = args.N_TEST+args.N_TRAIN
args.CLASSES = [[0], [6]]
args.NORMALIZED = True
args.NUM_TRIALS = 2
args.N_SAMPLES = [1024]
args.GAMMA = [0.1, 1, 10]
```

## Iterator specification

Now that the arguments are set, let's make sure our iterators are set up properly. For this, you'll need to make sure the number of samples is taken as iterator 0, and the number of trials is taken as iterator 1; otherwise, the iterators are fine to be dynamically set!

```python

iterators = [args.N_SAMPLES, range(args.NUM_TRIALS), args.GAMMA]
iterator_names = ["ntrain", "trial", "GAMMA"]
```
    
## Data selection

Load in the data here! Hopefully, I don't need to explain this...

## Batch function selection

A large portion of *MLPscape* is based off of batch functions. The batch functions should be set up in a particular way, like

```python
def your_bfn(X_total, y_total, X=None, y=None, bsz=128,
                     gen=None, **kwargs):
    def batch_fn(step: int, X=X, y=y):
        if (X is not None) and (y is not None):
            X = ensure_torch(X)
            y = ensure_torch(y)
            return X, y
        with torch.no_grad():
            N_total = X_total.shape[0]
            indices = torch.randint(0, N_total, (bsz,), generator=gen, device=gen.device)
            X_batch = ensure_torch(X_total.to(gen.device)[indices])
            y_batch = ensure_torch(y_total.to(gen.device)[indices])
            return X_batch, y_batch
    return batch_fn
```

which should be placed outside any "if __name__ == "__main()__:" statements so it can be found by _MLPscape_. Make sure kwargs is defined! Once this is set up, make sure bfn_config is set up to have

```python
other_args = dict(X_total = X_full, y_total = y_full) #one example
bfn_config = dict(base_bfn=your_bfn, *other_args)
```

## Other pretraining setup

We're almost to running the trainloop, we just need to include a global_config that gets used everywhere, which roughly takes the form of:

```python
global_config = dict(DEPTH=args.DEPTH, WIDTH=args.WIDTH, LR=args.LR, GAMMA=args.GAMMA,
    EMA_SMOOTHER=args.EMA_SMOOTHER, MAX_ITER=args.MAX_ITER,
    LOSS_CHECKPOINTS=args.LOSS_CHECKPOINTS, N_TEST=args.N_TEST,
    SEED=args.SEED, ONLYTHRESHOLDS=args.ONLYTHRESHOLDS, DIM=dim,
    ONLINE=args.ONLINE, VERBOSE=args.VERBOSE
    )
```

and also we'll want to make sure we get any grabs throughout the trainloop:

```python
grabs.update({"my_mlp_grab": my_mlp_grab})
global_config.update({"otherreturns": grabs})
```

Quick note:

```python
grabs = build_other_grabs(args.other_model_grabs, per_alias_kwargs=args.other_model_kwargs,)
```
is often included in my examples, but is not strictly necessary; only include if grabs aren't defined within your file.

## Trainloop execution

Finally, we can run everything! If using multiprocessing, call

```python
mp.set_start_method("spawn", force=True)
```

**Note**: If using a python notebook, the multiprocessing step doesn't play nice with an in-notebook batch function. Please either place the function in a .py file, or don't use multiprocessing.

Then, we call

```python    
result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config)
torch.cuda.empty_cache()
```

which is *really* simple!

I hope this helps in your model experimenting future!