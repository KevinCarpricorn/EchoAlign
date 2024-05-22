# Can We Treat Noisy Labels as Accurate?

This repository is the official implementation
of [Can We Treat Noisy Labels as Accurate?](https://arxiv.org/abs/2030.12345).

> Noisy labels significantly hinder the accuracy and generalization of machine learning models, particularly due to
> ambiguous instance features. Traditional techniques that attempt to correct noisy labels directly, such as those using
> transition matrices, often fail to address the inherent complexities of the problem sufficiently. In this paper, we
> introduce EchoAlign, a transformative paradigm shift in learning from noisy labels. Instead of focusing on label
> correction, EchoAlign treats noisy labels ($\tilde{Y}$) as accurate and modifies corresponding instance features ($X$)
> to achieve better alignment with $\tilde{Y}$. EchoAlign's core components are (1) EchoMod: Employing controllable
> generative models, EchoMod precisely modifies instances while maintaining their intrinsic characteristics and ensuring
> alignment with the noisy labels. (2) EchoSelect: Instance modification inevitably introduces distribution shifts between
> training and test sets. EchoSelect maintains a significant portion of clean original instances to mitigate these shifts.
> It leverages the distinct feature similarity distributions between original and modified instances as a robust tool for
> accurate sample selection. This integrated approach yields remarkable results. In environments with 30\%
> instance-dependent noise, even at 99\% selection accuracy, EchoSelect retains nearly twice the number of samples
> compared to the previous best method. Notably, on three datasets, EchoAlign surpasses previous state-of-the-art
> techniques with a substantial improvement.

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



