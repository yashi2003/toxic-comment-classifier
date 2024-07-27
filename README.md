# Toxic-comment-classifier
## NLP multi-label classification using Tensorflow

<div align="center">
  <img src="https://github.com/AlaGrine/Toxic-Comment-Classification-with-Tensorflow/blob/main/stats/toxic_comments_freepik.jpg" >
</div>

### Table of Contents

1. [Project Overview](#overview)
2. [Problem statement](#prbStatement)
3. [Installation](#installation)
4. [File Descriptions](#file_descriptions)
5. [Evaluation metrics and handling unbalanced dataset](#evaluation)
6. [Modelling](#modelling)
7. [Results](#results)
8. [Improvements](#Improvements)
9. [Acknowledgements](#Acknowledgements)

## Project Overview <a name="overview"></a>

Negative online behaviors, like toxic comments, are likely to make people stop expressing themselves and leave a conversation.

Platforms struggle to identify and flag potentially harmful or offensive online comments, leading many communities to restrict or shut down user comments altogether.

## Problem statement <a name="prbStatement"></a>

[Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) issued a challenge to build a multi-label classification model that's able to detect different types of toxicity like threats, obscenity and insults, and thus help make online discussion more productive and respectful.

The data for the problem is a dataset of 159,571 comments from Wikipedia’s talk page edits. These comments have been flagged for toxic behaviour by human reviewers.

## Installation <a name="installation"></a>

This project requires Python 3 and the following Python libraries installed:

`tensorflow` ,`pandas`, `numpy`, `scikit-multilearn`, `wordcloud`, `seaborn`, `matplotlib`, `plotly` and `googletrans`

## File Descriptions <a name="file_descriptions"></a>

The main file of the project is `Toxic-Comment-Classification-with-Tensorflow.ipynb`.

The project folder also contains the following:

- `stats` folder: The history and metrics of our models are available here (csv files).
- `googletrans` folder: contains the output of the text data augmentation with [googletrans](https://pypi.org/project/googletrans/) API.

## Evaluation metrics and handling unbalanced dataset <a name="evaluation"></a>

To deal with class imbalance, we:

> 1.  Used `stratified` data split instead of the traditional train_test_split.
> 2.  Performed `text data augmentation` to increase the number of minority labels (namely 'threat' and 'identity_hate'). We used the [googletrans](https://pypi.org/project/googletrans/) API to translate the given comment into French, German and Spanish and then back into English (back-translation).
> 3.  Built a `custom binary cross-entropy loss function with class weights`. The weights are inversely proportional to the number of samples in each class, so that the minority labels are given a greater weight.
> 4.  Used the **precision**, **recall** and **AUC** as evaluation metrics instead of the accuracy.

## Modelling <a name="modelling"></a>

We used `Tensorflow` to build the following model structure:

```
Input (text) -> TextVectorization -> Embedding -> custom_Layers -> Output (sigmoid)
```

where the `custom_Layers` component is either Dense Layer, LSTM, GRU or 1D Convolutional Neural Network.

## Results<a name="results"></a>

<div align="center">
  <img src="https://github.com/AlaGrine/Toxic-Comment-Classification-with-Tensorflow/blob/main/stats/AUC_dense_model.png" >
</div>

Applying our `custom binary cross-entropy loss function with class weights` and `text data augmentation` (using googletrans API) resulted in a **4%** increase in AUC for the dense model.

Both techniques were used for the other models.

<div align="center">
  <img src="https://github.com/AlaGrine/Toxic-Comment-Classification-with-Tensorflow/blob/main/stats/AUC_per_model.png" >
</div>

The `GRU` model performed best with an AUC of **97.7%**, **0.3%** ahead of the `LSTM` model.

I wrote a blog post about this project. You can find it [here](https://medium.com/@alaeddine.grine/toxic-comment-classification-317628632336).

## Improvements <a name="Improvements"></a>

We have been building our own embedding layer from scratch.

To improve our models, we could try different configurations, such as adding more layers or adjusting the number of neurons per layer.

Another alternative is to take advantage of `Transfer Learning` and use a pre-trained embedding layer, such as [BERT](https://tfhub.dev/google/collections/bert/1) available on the TensorFlow Hub.

## Acknowledgements <a name="Acknowledgements"></a>

Must give credit to [Kaggle](https://kaggle.com) for the dataset.
