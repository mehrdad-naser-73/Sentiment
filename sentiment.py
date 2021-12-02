import torch
from transformers import BertTokenizer, BertModel, AutoConfig, AutoTokenizer, AutoModel, AdamW
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import  flask
from flask import  Flask , request , Response
from flask_restful import Api , Resource

from model.bert_model import BertClassifier
from data.utils import Data, load_file


class sentiment_classifier:

    def __init__(self, device, n_class, model_name, lr):

        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.n_class = n_class
        self.model_name = model_name
        self.lr = lr
        self.save_model_path = "./" + self.model_name + "_best"
    def initialize(self):

        self.bert_classifier = BertClassifier(self.n_class, self.model_name)

        self.bert_classifier.to(self.device)
        self.params = list(self.bert_classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [p for n, p in self.params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': self.lr,
                'ori_lr': self.lr
            },
            {
                'params': [p for n, p in self.params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.lr,
                'ori_lr': self.lr
            }
        ]
        self.optimizer = AdamW(grouped_params, correct_bias=False)

    def train(self, train_dataloader, val_dataloader, epochs, evaluation):

        print("Start training...\n")

        best_val = -1
        for epoch_i in range(epochs):

            total_loss = 0
            self.bert_classifier.train()
            for batch in tqdm(train_dataloader):

                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                self.bert_classifier.zero_grad()

                logits = self.bert_classifier(b_input_ids, b_attn_mask)

                loss = self.loss_fn(logits, b_labels.view(-1))

                total_loss += loss.item()

                loss.backward()

                self.optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)

            if evaluation:
                val_accuracy = self.evaluate(val_dataloader)
                if val_accuracy > best_val:
                    print("saving best model....")
                    best_val = val_accuracy
                    torch.save(self.bert_classifier.state_dict(), self.save_model_path)
                print("Acc for Validation set: ", val_accuracy, "\n")

        print("Training complete!")

    def evaluate(self, d_loader):

        self.bert_classifier.eval()
        pred_list = []
        true_list = []

        for batch in d_loader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                logits = self.bert_classifier(b_input_ids, b_attn_mask)

            preds = torch.argmax(logits, dim=1).flatten()
            pred_list += preds.cpu().numpy().tolist()
            true_list += b_labels.view(-1).cpu().numpy().tolist()

        pred_list, true_list = np.array(pred_list), np.array(true_list)
        accuracy = np.sum(pred_list == true_list) / len(pred_list)

        return accuracy

    def load_model(self, path=None):

        if path==None:
            path = self.save_model_path
        print("loading model from "+ path)
        self.bert_classifier.load_state_dict(torch.load(path))
        print("model loaded!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='parsbert',
                        help='"parsbert" for ParsBERT and "multi" for Multilingual BERT')
    parser.add_argument('--max_len', default=64,
                        help='maximum sequence length')
    parser.add_argument('--batch_size', default=32,
                        help='batch size')
    parser.add_argument('--lr', default=1e-5,
                        help='learning rate')
    parser.add_argument('--mode', default="train",
                        help='"train", "test" or "webservice"')
    parser.add_argument('--load_model_path', default="",
                        help='path of the model used for testing or webservice')
    parser.add_argument('--data_dir', default="./dataset",
                        help='path of the dataset directory')
    parser.add_argument('--epochs', default=2,
                        help='number of training epochs')
    parser.add_argument('--num_classes', default=2,
                        help='number of classes')

    args = parser.parse_args()
    ### Load data
    train_t, train_l = load_file(args.data_dir+'/train.csv')

    ### Training set is too large! takes a lot of time to train with the colab gpu
    _, train_t, _, train_l = train_test_split(train_t, train_l, test_size=0.33, random_state=42, stratify=train_l)

    dev_t, dev_l = load_file(args.data_dir+'/dev.csv')
    test_t, test_l = load_file(args.data_dir+'/test.csv')

    ###
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    max_len = args.max_len
    batch_size = args.max_len
    
    if args.model_name == 'parsbert':
        pretrain_path = "HooshvareLab/bert-base-parsbert-uncased"
        tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    elif args.model_name == 'multi':
        pretrain_path = 'bert-base-multilingual-cased'
        tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    d_train, d_dev, d_test = Data(train_t, train_l, max_len, tokenizer, batch_size), Data(dev_t, dev_l, max_len, tokenizer, batch_size), Data(test_t, test_l, max_len, tokenizer, batch_size)

    train_dataloader, val_dataloader, test_dataloader  = d_train.get_data_loader(mode="train"), d_dev.get_data_loader(mode="dev"), d_test.get_data_loader(mode="test")

    classifier = sentiment_classifier(device=device,
                                               n_class=args.num_classes,
                                               model_name=args.model_name,
                                               lr=args.lr)
    classifier.initialize()
    if args.mode == "train":
        classifier.train(train_dataloader, val_dataloader, epochs=args.epochs, evaluation=True)
        classifier.load_model(classifier.save_model_path)
        test_acc = classifier.evaluate(test_dataloader)
        print("Acc for test set: ", test_acc, "\n")

    elif args.mode == "test":
        classifier.load_model(args.load_model_path)
        test_acc = classifier.evaluate(test_dataloader)
        print("Acc for test set: ", test_acc, "\n")

    elif args.mode == "webservice":
        classifier.load_model(args.load_model_path)
        app = Flask(__name__)
        api = Api(app)

        class SA(Resource):

            def get(self):

                sentence = request.args.get('text', '')
                d_test = Data([sentence], ['1'], max_len, tokenizer, 1)
                test_dataloader = d_test.get_data_loader(mode="test")
                for batch in test_dataloader:
                    b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                    with torch.no_grad():
                        logits = classifier.bert_classifier(b_input_ids, b_attn_mask)
                    preds = torch.argmax(logits, dim=1).flatten()
                sentiment = "Happy" if preds[0]==0 else "Sad"

                return {"sentiment": sentiment}


        api.add_resource(SA, "/")
        app.run()

                

