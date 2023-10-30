# Machine learning code for Natural Language Processing

This is a model that uses the concepts of bigrams and trigrams in sentiment analysis of movie reviews.
Following which, we can investigate the effect of bigrams compared to trigrams in the accuracy of analysing sentiment analysis.
You should have python3 installed in your local machine.

## How to use
1. Clone the repo into local machine
```shell
git clone https://github.com/Chustinjeng/IT1244-project
```

2. Go into the directory of IT1244-project


3. To train the model, in your terminal, type 
```shell
python3 sentiment.py --train --text_path x_train.txt --label_path y_train.txt --model_path model.pt --bigram 
```
if you want to train the model using bigrams

```shell
python3 sentiment.py --train --text_path x_train.txt --label_path y_train.txt --model_path model.pt --trigram 
```
if you want to train the model using trigrams


4. To test the model on the accuracy, in your terminal, type
```shell
python3 sentiment.py --test --text_path x_test.txt --model_path model.pt --output_path out.txt --bigram
```
if you want to test the model using bigrams 

```shell
python3 sentiment.py --test --text_path x_test.txt --model_path model.pt --output_path out.txt --trigram
```
if you want to test the model using trigrams

***Please choose the same type across training and testing! Otherwise the accuracy will plummet :sad:***


5. To calculate the accuracy of the model, in your terminal, type
```shell
python3 eval.py out.txt y_test.txt
```