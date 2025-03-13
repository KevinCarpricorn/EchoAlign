# Can We Treat Noisy Labels as Accurate?

This repository is the official implementation
of [Can We Treat Noisy Labels as Accurate?](https://arxiv.org/abs/2405.12969).

>Noisy labels significantly hinder the accuracy and generalization of machine learning models, particularly when resulting from ambiguous instance features that complicate correct labeling. Traditional approaches, such as those relying on transition matrices for label correction, often struggle to effectively resolve such ambiguity, due to their inability to capture complex relationships between instances and noisy labels. In this paper, we propose EchoAlign, a paradigm shift in learning from noisy labels. Unlike previous methods that attempt to correct labels, EchoAlign treats noisy labels ($\tilde{Y}$) as accurate and modifies corresponding instances ($X$) to better align with these labels. The EchoAlign framework comprises two main components: (1) EchoMod leverages controllable generative models to selectively modify instance features, achieving alignment with noisy labels while preserving intrinsic instance characteristics such as shape, texture, and semantic identity. (2) EchoSelect mitigates distribution shifts introduced by instance modifications by strategically retaining a substantial subset of original instances with correct labels. Specifically, EchoSelect exploits feature similarity distributions between original and modified instances to accurately distinguish between correctly and incorrectly labeled samples. Extensive experiments across three benchmark datasets demonstrate that EchoAlign significantly outperforms state-of-the-art methods, particularly in high-noise environments, achieving superior accuracy and robustness. Notably, under 30\% instance-dependent noise, EchoSelect retains nearly twice the number of correctly labeled samples compared to previous methods, maintaining 99\% selection accuracy, thereby clearly illustrating the effectiveness of EchoAlign.

## Requirements

The project is implemented in PyTorch. To install the required libraries, run the following command:

```bash
conda env create -f environment.yml
conda activate EchoAlign
```

## Training

To train the models in the project, use the following command:

```bash
python main.py --dataset cifar10 --num_classes 10 --noise_type instance --noise_rate 0.5 --threshold 0.52
python main.py --dataset cifar100 --num_classes 100 --noise_type instance --noise_rate 0.5 --threshold 0.52
python main.py --dataset cifar10N --num_classes 10 --noise_type real --real_type random_label1 --threshold 0.43
```

> 📋 The `main.py` script contains the main training loop and uses the parsed arguments from `args_parser.py`.

## License and Contributing

- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github.

### Reference

If you find the code useful in your research, please consider citing our paper:
```
@misc{zheng2024echoalign,
      title={Can We Treat Noisy Labels as Accurate?}, 
      author={Yuxiang Zheng and Zhongyi Han and Yilong Yin and Xin Gao and Tongliang Liu},
      year={2024},
      eprint={2405.12969},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.12969}, 
}
```


