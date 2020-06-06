# Machine Translation with Transformers
  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The goal of this repository is to show how to implement a machine translation system using the nn.Transformer class from Pytorch and the HuggingFace library. If you are familiar with the state of the art in NLP, you might have read the original paper [Attention is all you need](https://arxiv.org/abs/1706.03762). It is a fantastic architecture and I encourage you to take a look and understand the architecture before looking at this code. I have used multiple references around the internet to come up with this single piece of code and I may have referred to different code repositories to borrow few implementations. For example, for calling the `nn.Transformer` class itself, I have borrowed the syntax from [this](https://github.com/andrewpeng02/transformer-translation) excellent repository.

## DEPENDENCIES:
This code has been written using **Python 3.7**
* torch == 1.5.0
* einops == 0.2.0
* tqdm == 4.46.1
* transformers == 2.11.0
* numpy

You can even create a Pipenv virtual environment using the Pipfile provided by running

```python
pipenv install .
```

## NOTE:

The objective of this repository is **NOT** to provide a state of the art machine translation model. This is **NOT** production level code. Instead the goal is to provide you a tutorial on how to train a machine translation model in PyTorch using HuggingFace library and specifically the [nn.Transformer](https://pytorch.org/docs/master/generated/torch.nn.Transformer.html) class in PyTorch :relaxed: . I have used the [HuggingFace](https://github.com/huggingface/transformers) for tokenizing the text.

## INPUT DATA:

Preparing the input data is very simple. Suppose you are going to train a machine translation system to translate from English to French, you need two text files, one for English and the other for French. Each of these files have on every line a sentence from English in the file corresponding to English and the corresponding sentence from French in the file corresponding to French. You need to have two of these pairs, one for training and the other for validation.

Check the structure of the files in the `data/raw/en` and the `data/raw/fr` folders that contains sample text files if you want to use them. 

## INSTALLATION & USAGE

Clone the repo 

```bash
git clone hhttps://github.com/abhisheksgumadi/machine-translation-transformers.git
cd machine-translation-transformers
```

Once you are inside the main directory, run
```python
python train.py
``` 
Have fun! and play around with the code to understand how to train a machine translation model by fine tuning the `nn.Transformer` model in PyTorch and also using HuggingFace :blush:. Change the codebase hovewer you want, experiment with it and most importantly learn something new today :smile:
