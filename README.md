# PubMed Abstract Classifier

This repository contains the code implementation of the text classification model used in the following paper:  
**PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts**  
The original paper can be found at:  
`https://arxiv.org/pdf/1710.06071.pdf`

# Dataset

PubMed RCT (PubMed 200k RCT)  

**The dataset consists of approximately 200,000 abstracts of randomized controlled trials, totaling 2.3 million sentences**. Each sentence of each abstract is labeled with their role in the abstract using one of the following classes: background, objective, method, result, or conclusion.  

You can download the dataset from the following link:
`https://github.com/Franck-Dernoncourt/pubmed-rct`

Or you can simply clone it in the project using:  
`git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git`  

## Model 

The model, I created is similar to the one in original paper but with slight changes, such as:  
	 *1. I created and used my own custom token and character embeddings in the model.*  
	 *2. In paper, the optimizer used was SGD, whereas i used Adam optimizer.*  
   *3. Dropout rate of 50% is applied to reduce over-fitting.*  
	 *4. Label smoothing of 0.2 is applied, so our model can generalize well.*
   
**Model Architecture**  
![tribrid model architecture](https://user-images.githubusercontent.com/111675443/221554544-f3ea2693-c3a0-4129-a6cf-e363926b6ba0.png)

## Clone this repository

`git clone https://github.com/r-zeeshan/pubmed-abstract-classifier.git`  

## Colab NoteBook

This repository also contains the colab notebook version. You can directly open this notebook in Google Colab, and the run cell by cell if you want to train this model for your self.

  **Thank You**
