# Simple, Good, Fast (SGF)

Official code of the paper [Simple, Good, Fast: Self-Supervised World Models Free of Baggage](https://openreview.net/forum?id=yFGR36PLDJ).  
Published as a conference paper at ICLR 2025.

If you find this code or paper helpful, please reference it using the following citation:

```
@inproceedings{
  robine2025simple,
  title={Simple, Good, Fast: Self-Supervised World Models Free of Baggage},
  author={Jan Robine and Marc H{\"o}ftmann and Stefan Harmeling},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=yFGR36PLDJ}
}
```

## Setup

Make sure the following dependencies are installed:  
`torch`, `torchvision`, `gymnasium`, `numpy`, `wandb`, `ruamel.yaml`  
Tested with PyTorch 2.5.1.

## Training

To start a training run, execute the following command:

```bash
$ python src/main.py --device cuda:0 --game Breakout --project sgf --config configs/default.yaml --amp --compile --seed 1
```

The training script will log all relevant information to Weights & Biases.

To change the hyperparameters, create a copy of the `default.yaml` file and adjust the values as needed.

You can speed up training by setting `--agent_eval final`, which will only evaluate the agent at the end of training. To train an additional decoder for debugging, set `--wm_eval decoder`.
