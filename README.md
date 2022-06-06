# HydraNets Subtask Partitioning

This code implements the subtask partitioning algorithm described by the
[HydraNets](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mullapudi_HydraNets_Specialized_Dynamic_CVPR_2018_paper.pdf)
paper (Mullapudi, et al). It was originally designed to partition the ImageNet
dataset.

To set up the virtual environment, run:
```bash
$ python3 -m venv <path/to/virtualenv>
$ source <path/to/virtualenv/>/bin/activate
```

## Algorithm Overview

At a high level, the HydraNets algorithm to partition a superset of size `C`
into `n` subsets is as follows:
1. For each class, compute a feature representation by taking the average of the
features from the final fully connected layer of an image classification network
for several images of that class.
2. Use k-means to find `n` centroids for `n` clusters.
3. For each of the `n` centroids, assign the `C / n` closest classes to it (to
handle the generalized case where `C mod n != 0`, just assign the remaining
amount of classes if that amount is less than `C/ n`).

The paper covers the algorithm in its entirety in Section 3.1.

## Code Setup

If your local repository already has a virtual environment, you skip the
creation step and just activate that one.

After setting up your virtual environment, install the required dependencies by
running:
```bash
$ pip3 install -r requirements.txt
```


