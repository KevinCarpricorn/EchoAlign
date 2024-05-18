# Can We Treat Noisy Labels as Accurate?

This repository is the official implementation
of [Can We Treat Noisy Labels as Accurate?](https://arxiv.org/abs/2030.12345).

> In machine learning, noisy labels undermine models' accuracy and generalization capability due to ambiguous instance
> features. Traditional methods, such as using transition matrices for label correction, have not effectively resolved the
> complexities associated with noisy labels. Challenging this direct correction approach, this paper introduces a
> transformative paradigm—EchoAlign, which reinterprets noisy labels ($\tilde{Y}$) as accurate and adjusts the
> corresponding instance features ($X$) to align with $\tilde{Y}$. EchoAlign employs advanced generative modeling to
> modify instances, ensuring minimal style deviation from their original state while preserving label integrity. EchoAlign
> introduces EchoFilter, an innovative sample selection technique that effectively mitigates distribution shifts between
> training and testing datasets, adeptly handling both covariate and label shifts. This methodology conserves instances'
> intrinsic properties and enhances their alignment with designated labels, substantially improving model performance.
> Through EchoAlign, we demonstrate significant advancements in learning from noisy labels, achieving up to 98\% accuracy
> in correcting instance-dependent noise. Remarkably, on the CIFAR-10 dataset with an instance-dependent noise rate of
> 0.5, EchoAlign outperforms state-of-the-art methods by an impressive 7\% increase in accuracy.

## Requirements

The project is implemented in PyTorch. To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Training

To train the models in the project, use the following command:

```bash
python main.py --dataset cifar10 --num_classes 10 --noise_type instance --noise_rate 0.5 
```

> 📋 The `main.py` script contains the main training loop and uses the parsed arguments from `args_parser.py`.

## License and Contributing

- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github.

### Reference

If you find the code useful in your research, please consider citing our paper:

```latex
@inproceedings{
    2024echoalign,
    title={Can We Treat Noisy Labels as Accurate?},
    author={},
    booktitle={NeurIPS},
    year={2024},
}
```

