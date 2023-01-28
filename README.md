# 'Algorithm Design and Analysis' project——Chinese named entity recognition (21spring)

Xianlin Zhao  Jiebin Zhang



## Approaches

This is a course project. We tried some approaches for Chinese NER problem.

* **HMM**
* modified HMM (with the idea of **MEMM**)
* **CRF** (Conditional Random Field)    use "sklearn-crfsuite"
* **BiLSTM**
* **BiLSTM + CRF**
* **Lattice LSTM** (ACL 2018 paper: Chinese NER Using Lattice LSTM. https://github.com/jiesutd/LatticeLSTM  We did some modifications to adapt to Python and PyTorch version. hyperparameters can be changed in ``./utils/data.py``)
* **Transformer** (2 encoder layers, 4 heads)
* **Self-Attention + BiLSTM**

## Dataset

``./data/renMinRiBao``

BIO labels: O, B-LOC, B-PER, B-ORG, B-DATE, I-LOC, I-PER, I-ORG, I-DATE.

## Requirement

```shell
Python 3.7
PyTorch 1.7.1
NumPy 1.19.2
sklearn-crfsuite 0.3.6
```

## Reference

https://www.cnblogs.com/en-heng/p/6201893.html

https://sklearn-crfsuite.readthedocs.io/en/latest/api/[html](https://sklearn-crfsuite.readthedocs.io/en/latest/api/html)

https://www.bilibili.com/video/BV1JE411g7XF?p=20 

https://arxiv.org/pdf/1805.02023.pdf

https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

https://createmomo.github.io/2017/11/11/CRF-Layer-on-the-Top-of-BiLSTM-5/

https://arxiv.org/pdf/1706.03762.pdf

https://www.cnblogs.com/bep-feijin/articles/9841645.html

https://zhuanlan.zhihu.com/p/33397147

