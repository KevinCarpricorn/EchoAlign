# Input Harmony: Revolutionizing Noisy Label Learning with ControlNet-Driven Feature Adaptation

This repository is the official implementation of [Input Harmony: Revolutionizing Noisy Label Learning with ControlNet-Driven Feature Adaptation](https://arxiv.org/abs/2030.12345). 

>In the realm of machine learning, learning with noisy labels is a pervasive challenge that often impairs the performance of supervised learning algorithms. Traditional approaches predominantly focus on label correction strategies, utilizing techniques like transition matrices to amend noisy labels (Y). However, these strategies inherently fall short of fully rectifying the label inaccuracies. To the best of our knowledge, there exists a significant gap in the literature concerning the adjustment of input features (X) to complement noisy labels. This paper introduces a novel paradigm, leveraging a technique named ControlNet, to modify the input (X) in alignment with the noisy labels (Y), potentially achieving a more comprehensive correction. ControlNet, inspired by the diffusion model and infused with control information, has garnered acclaim as the best paper in this year's International Conference on Computer Vision (ICCV). It shows promising potential in addressing the label noise issue. Our empirical results already surpass the current state-of-the-art (SOTA) performances.

## Requirements

The project is implemented in PyTorch. To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Training

To train the models in the project, use the following command:

```bash
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5 
```

>📋 The `main.py` script contains the main training loop and uses the parsed arguments from `args_parser.py`.

## License and Contributing

- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github.

### Reference

If you find the code useful in your research, please consider citing our paper:

```latex
@inproceedings{
    zheng2024,
    title={Input Harmony: Revolutionizing Noisy Label Learning with ControlNet-Driven Feature Adaptation},
    author={...},
    booktitle={...},
    year={2024},
}
```

