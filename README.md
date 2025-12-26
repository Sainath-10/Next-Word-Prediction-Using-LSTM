# Next-Word-Prediction-Using-LSTM
LSTM Based Deep Learning Model To Predict The Next Word Using Hamlet by William Shakespeare

# 🎭 Shakespeare Next-Word Prediction using LSTM

A Deep Learning project that uses **Natural Language Processing (NLP)** and **LSTM (Long Short-Term Memory)** networks to predict the **next word** in a sentence based on Shakespeare’s *Hamlet* text.

---

## 📘 Overview

This project demonstrates how **sequence modeling** works in NLP using **Recurrent Neural Networks (LSTM)**.
The model learns Shakespeare’s writing style by predicting the next word given a sequence of previous words.

It involves:

* Text preprocessing and tokenization
* Creating n-gram input sequences
* Padding sequences for uniform length
* Training an LSTM-based language model
* Predicting the next word for a given input sentence

---

## 📂 Dataset Description

* **Source**: NLTK Gutenberg Corpus  
* **Text**: *Hamlet* by William Shakespeare  
* **Type**: Raw literary text   

The text is tokenized and converted into numerical sequences for training.

---

## ⚙️ Tech Stack

* **Language**: Python  
* **Libraries**: TensorFlow / Keras, NumPy, NLTK, Streamlit  

---

## 🧩 Project Workflow

### 1. Data Collection
Shakespeare’s *Hamlet* text is loaded from the NLTK Gutenberg corpus and stored locally.

### 2. Text Preprocessing
* Tokenization using Keras Tokenizer  
* Conversion of text into word indices  
* Creation of n-gram sequences  

### 3. Sequence Preparation
* Padding sequences to equal length  
* Splitting data into input (X) and target (y)  

### 4. Model Building
Deep Learning model using:
* Embedding Layer  
* Stacked LSTM layers  
* Dropout for regularization  
* Dense Softmax output layer  
---

## 📊 Results

**Training Accuracy: ~75%**

Accuracy is expected to be moderate due to:
* Large vocabulary size  
* Multiple valid next-word possibilities  

