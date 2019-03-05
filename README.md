# News Category Classification with BERT
Identify the type of news based on headlines and short descriptions

# Dataset
This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. The model trained on this dataset could be used to identify tags for untracked news articles or to identify the type of language used in different news articles. [Kaggle](https://www.kaggle.com/rmisra/news-category-dataset)
  
# Implementations
  - [x] BERT (Fine-Tuning)
  - [x] Bi-GRU + CONV
  - [x] LSTM + Attention
  
## Try it on [Colab Notebook](https://colab.research.google.com/drive/1wPXAuNP-iXsXxBhxG0l6Yv94gqlGNPJB)

# TL;DR
  
* [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip) (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) was used as the embedding layer for the Bi-GRU and LSTM models.
* bert-base-uncased (12-layer, 768-hidden, 12-heads, 110M parameters) pre-trained model was used.

# Resuts
- `BERT` - test_accuracy: 1.0, test_loss: 0.0015671474330127238
- `Bidirectional GRU + Conv` - test_accuracy: 0.6545
- `LSTM with Attention` - test_accuracy: 0.67144

# Requirements

* Python 3.6 
* PyTorch 0.4.1/1.0.0 - For the creation of BiLSTM-CRF architecture
* pytorch-pretrained-bert - https://github.com/huggingface/pytorch-pretrained-BERT


