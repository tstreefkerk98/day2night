# day2night

### Logging

Weight and Biases (wandb) is used to log the training sessions. To use this, rename `env.example.py` to `env.py` and
replace the required fields with your own credentials.

### Training set

Day-time data used during training: CityScapes <br>
Night-time data used during training: DarkZurich

For training, add CityScapes to `data/train/day` and DarkZurich to `data/train/night`.

### Training

To train the model run the at least following command:

`python train.py [--cycle_wgan|--cycle_gan]`

To view more options, run `python train.py -h`

### References

A large part of the code has been copied from these sources and altered to fit the day-night domain.

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```

```
@article{hu_improved_nodate,
	title = {Improved {CycleGAN} for Image-to-Image Translation},
		pages = {9},
	author = {Hu, Weining and Li, Meng and Ju, Xiaomeng},
	langid = {english}
}
```

The CIConv layer (`ciconv2d.py`) is a slightly altered version of the original CIConv layer from:
```
@article{lengyel_zero-shot_2021,
	title = {Zero-Shot Day-Night Domain Adaptation with a Physics Prior},
	url = {http://arxiv.org/abs/2108.05137},
	journaltitle = {{arXiv}:2108.05137 [cs]},
	author = {Lengyel, Attila and Garg, Sourav and Milford, Michael and van Gemert, Jan C.},
	urldate = {2021-11-18},
	date = {2021-10-11},
	eprinttype = {arxiv},
	eprint = {2108.05137},
	keywords = {Computer Science - Computer Vision and Pattern Recognition},
}
```