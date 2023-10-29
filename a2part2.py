import os
import re
import sys
import string
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class SentDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """
    def __init__(self, train_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            vocab (string): Path to the vocabulary file.
            text_file (string): Path to the text file.
        """
        self.label_path = label_path
        self.texts = []
        self.labels = []
        with open(train_path, encoding='utf-8') as f:
            self.texts = [line for line in f.readlines() if line.strip()]
        if label_path:
            with open(label_path, encoding='utf-8') as f:
                self.labels = [line for line in f.readlines() if line.strip()]
        if not vocab:
            self.vocabulary = {}
            curr_idx = 0
            for text in self.texts:
                bigrams = self.generate_bigrams(text)
                for bigram in bigrams:
                    if bigram in self.vocabulary:
                        continue
                    else:
                        self.vocabulary[bigram] = curr_idx
                        curr_idx += 1
        else: 
            self.vocabulary = vocab
        

    def generate_bigrams(self, text):
        tokens = text.split()
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            bigrams.append(bigram)
        return bigrams


    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
        """
        return len(self.vocabulary)

    
    
    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        if self.label_path:
            text = self.texts[i]
            label = int(self.labels[i])
            indices = []
            bigrams_in_text = self.generate_bigrams(text)
            for bigram in bigrams_in_text:
                index = self.vocabulary.get(bigram)
                indices.append(index)
    
            indices_tensor = torch.tensor(indices)
            return indices_tensor, label
        
        else:
            text = self.texts[i]
            indices = []
            bigrams_in_text = self.generate_bigrams(text)
            # print(bigrams_in_text)
            # print("vocab: ", self.vocabulary)
            for bigram in bigrams_in_text:
                if bigram in self.vocabulary:
                    index = self.vocabulary.get(bigram)
                    indices.append(index)
            # print("indices:", indices)
            indices_tensor = torch.tensor(indices)
            return indices_tensor


class Model(nn.Module):
    """
    Define your model here
    """
    def __init__(self, num_vocab):
        super().__init__()
        # define your model attributes here
        # print("num_vocab: ", num_vocab)
        self.embedding_dim = 8
        self.embedding = nn.Embedding(num_vocab, self.embedding_dim)
        self.first_layer_dim = 24
        self.linear_layer_1 = nn.Linear(self.embedding_dim, self.first_layer_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear_layer_2 = nn.Linear(self.first_layer_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # define the forward function here
        x = self.embedding(x)
        # print('embedding m: ', x)
        x = torch.mean(x, dim=1)
        x = self.linear_layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_layer_2(x)
        x = self.sigmoid(x)

        return x


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    if len(batch[0]) == 2:
        texts, labels = zip(*batch)
        # convert text indices to tensor
        texts_tensor = nn.utils.rnn.pad_sequence([text for text in texts], batch_first=True, padding_value=0)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return texts_tensor, labels_tensor
    else:
        # texts = zip(*batch)
        # print('texts: ', list(texts))
        texts_tensor = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        return texts_tensor


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    
    Do not calculate the loss from padding.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        # print("epoch: ", epoch)
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # do forward propagation
            outputs = model(texts)

            # calculate the loss
            loss = criterion(outputs, labels.reshape((outputs.shape[0], 1)))

            # do backward propagation
            loss.backward()

            # do the parameter optimization
            optimizer.step()

            # calculate running loss value for non padding
            running_loss += loss.item()

            # print loss value every 100 iterations and reset running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': num_epoch,
        'vocab': dataset.vocabulary
    }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, thres=0.5, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            # print("data: ", data)
            texts = data.to(device)
            results = model(texts)
            pred_labels = (results > thres).int().tolist()
            pred_labels = sum(pred_labels, [])
            # print('pred labels: ', pred_labels)
            labels.extend(pred_labels)

    return [str(x) for x in labels]


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        dataset = SentDataset(args.text_path, args.label_path)
        num_vocab = dataset.vocab_size()
        model = Model(num_vocab).to(device)
        
        # specify these hyper-parameters
        batch_size = 48
        learning_rate = 0.01
        num_epochs = 10

        train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        # load the checkpoint
        checkpoint = torch.load(args.model_path)

        # create the test dataset object using SentDataset class
        dataset = SentDataset(args.text_path, vocab=checkpoint["vocab"])

        # initialize and load the model
        num_vocab = dataset.vocab_size()
        model = Model(num_vocab).to(device)
        model.load_state_dict(checkpoint["model_state"])

        # run the prediction
        preds = test(model, dataset, 0.5, device)
        # print('predicted value: ', preds)
        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(preds))
    print('\n==== All done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the model file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)