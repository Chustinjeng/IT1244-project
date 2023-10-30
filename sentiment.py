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
    A pytorch dataset class that accepts a text path, and optionally label path (only during training phase) and
    a vocabulary (only during testing phase). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    """
    def __init__(self, train_path, type, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            vocab (string): Path to the vocabulary file.
            text_file (string): Path to the text file.
            type (string): Specify if model is trained using bigrams or trigrams
        """
        self.label_path = label_path
        self.type = type
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
                ngrams = self.generate_bigrams_or_trigrams(text, type)
                for ngram in ngrams:
                    if ngram in self.vocabulary:
                        continue
                    else:
                        self.vocabulary[ngram] = curr_idx
                        curr_idx += 1
        else: 
            self.vocabulary = vocab
        

    def generate_bigrams(self, text):
        """
        Function to generate bigrams from a text (string)
        Bigrams are defined as a grouping of a text into a list of 2 consecutive words
        """
        tokens = text.split()
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            bigrams.append(bigram)
        return bigrams
    
    def generate_trigrams(self, text):
        """
        Function to generate bigrams from a text (string)
        Bigrams are defined as a grouping of a text into a list of 3 consecutive words
        """
        tokens = text.split()
        trigrams = []
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
            trigrams.append(trigram)
        return trigrams
    
    def generate_bigrams_or_trigrams(self, text, type):
        """
        Function to determine if bigrams or trigrams should be generated, depending on type specified
        """
        if type == "bigram":
            return self.generate_bigrams(text)
        else: 
            return self.generate_trigrams(text)


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
        Text and label is encoded according to the vocab (word_id).

        """
        if self.label_path: # training
            text = self.texts[i]
            label = int(self.labels[i])
            indices = []
            ngrams_in_text = self.generate_bigrams_or_trigrams(text, self.type)
            for ngram in ngrams_in_text:
                index = self.vocabulary.get(ngram)
                indices.append(index)
    
            indices_tensor = torch.tensor(indices)
            return indices_tensor, label
        
        else: # testing 
            text = self.texts[i]
            indices = []
            ngrams_in_text = self.generate_bigrams_or_trigrams(text, self.type)
            for ngram in ngrams_in_text:
                if ngram in self.vocabulary:
                    index = self.vocabulary.get(ngram)
                    indices.append(index)
            indices_tensor = torch.tensor(indices)
            return indices_tensor


class Model(nn.Module):
    """
    Define your model here
    """
    def __init__(self, num_vocab):
        super().__init__()
        # define your model attributes here
        self.embedding_dim = 8 # define embedding dimensions (hyperparameter)
        self.embedding = nn.Embedding(num_vocab, self.embedding_dim) # transform words into embeddings
        self.first_layer_dim = 24 # define first layer dimension (hyperparameter)
        self.linear_layer_1 = nn.Linear(self.embedding_dim, self.first_layer_dim) # linear layer
        self.relu = nn.ReLU() # ReLU activation function
        self.dropout = nn.Dropout(0.2) # dropout of 0.2 probability (hyperparameter) to reduce overfitting
        self.linear_layer_2 = nn.Linear(self.first_layer_dim, 1) # last linear layer
        self.sigmoid = nn.Sigmoid() # Sigmoid function to determine probabilities


    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.linear_layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_layer_2(x)
        x = self.sigmoid(x)

        return x


def collator(batch):
    """
    A function that receives a list of (text, label) pair
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
        texts_tensor = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        return texts_tensor


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    
    """
    # instantiate the data loader which loads data in batches
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # loss function is Binary Cross Entropy Loss
    criterion = nn.BCELoss()
    # optimiser is Adam's optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
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

            # do backward propagation to update the weights
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
    
    # make the checkpoint of the model and save it to the model path
    # contains current state of the model, optimiser, number of epochs, and current vocabulary
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
            texts = data.to(device)
            results = model(texts)
            pred_labels = (results > thres).int().tolist()
            pred_labels = sum(pred_labels, [])
            labels.extend(pred_labels)

    return [str(x) for x in labels]


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.bigram or args.trigram, "Please specify --bigram or --trigram"
    assert not (args.bigram and args.trigram), "Please specify only --trigram or --bigram"
    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        if args.bigram:
            dataset = SentDataset(args.text_path, "bigram", args.label_path)
            num_vocab = dataset.vocab_size()
            model = Model(num_vocab).to(device)
            
            # specify hyper-parameters
            batch_size = 48
            learning_rate = 0.01
            num_epochs = 10

            train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
        else:
            dataset = SentDataset(args.text_path, "trigram", args.label_path)
            num_vocab = dataset.vocab_size()
            model = Model(num_vocab).to(device)
            
            # specify hyper-parameters
            batch_size = 48
            learning_rate = 0.01
            num_epochs = 10

            train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)

    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        # load the checkpoint
        checkpoint = torch.load(args.model_path)

        # create the test dataset object using SentDataset class
        dataset = None
        if args.bigram:
            dataset = SentDataset(args.text_path, "bigram" ,vocab=checkpoint["vocab"])
        else: 
            dataset = SentDataset(args.text_path, "trigram" ,vocab=checkpoint["vocab"])
            
        # initialize and load the model
        num_vocab = dataset.vocab_size()
        model = Model(num_vocab).to(device)
        model.load_state_dict(checkpoint["model_state"])

        # run the prediction
        preds = test(model, dataset, 0.5, device)
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
    parser.add_argument('--bigram', action='store_true', help='train model using bigrams')
    parser.add_argument('--trigram', action='store_true', help='train model using trigrams')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)