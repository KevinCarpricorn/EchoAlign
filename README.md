# Can We Treat Noisy Labels as Accurate?

This repository is the official implementation
of [Can We Treat Noisy Labels as Accurate?](https://arxiv.org/abs/2030.12345).

>Noisy labels present a significant challenge to the accuracy and generalization of machine learning models, especially when they originate from ambiguous instance features. Traditional methods for directly addressing noisy labels, such as those using transition matrices, often fail to adequately capture label noise caused by ambiguous features. In this paper, we propose EchoAlign, a paradigm shift in learning from noisy labels. Rather than correcting noisy labels, EchoAlign treats noisy labels ($\tilde{Y}$) as accurate and modifies the corresponding instance features ($X$) to achieve better alignment with $\tilde{Y}$. The EchoAlign framework has two primary components: 1) EchoMod utilizes controllable generative models to modify instances $X$ while preserving intrinsic characteristics. 2) EchoSelect mitigates distribution shifts by retaining a substantial portion of original, correctly labeled instances. EchoSelect further enhances selection accuracy by leveraging feature similarity distributions. Across three datasets, EchoAlign significantly outperforms state-of-the-art methods, especially in high-noise scenarios, showcasing both superior accuracy and robustness. In settings with 30\% instance-dependent noise, EchoSelect retains nearly twice as many samples as previous methods at 99\% selection accuracy, demonstrating the observed improvements.

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



