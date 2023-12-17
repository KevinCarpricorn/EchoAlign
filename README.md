# Input Harmony: Revolutionizing Noisy Label Learning with ControlNet-Driven Feature Adaptation

This repository is the official implementation of [Input Harmony: Revolutionizing Noisy Label Learning with ControlNet-Driven Feature Adaptation](https://arxiv.org/abs/2030.12345). 

>In the realm of machine learning, learning with noisy labels is a pervasive challenge that often impairs the performance of supervised learning algorithms. Traditional approaches predominantly focus on label correction strategies, utilizing techniques like transition matrices to amend noisy labels (Y). However, these strategies inherently fall short of fully rectifying the label inaccuracies. To the best of our knowledge, there exists a significant gap in the literature concerning the adjustment of input features (X) to complement noisy labels. This paper introduces a novel paradigm, leveraging a technique named ControlNet, to modify the input (X) in alignment with the noisy labels (Y), potentially achieving a more comprehensive correction. ControlNet, inspired by the diffusion model and infused with control information, has garnered acclaim as the best paper in this year's International Conference on Computer Vision (ICCV). It shows promising potential in addressing the label noise issue. Our empirical results already surpass the current state-of-the-art (SOTA) performances.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 