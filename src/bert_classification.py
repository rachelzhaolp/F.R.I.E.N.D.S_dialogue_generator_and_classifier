import warnings

warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
import sys
import logging
from os import path
import os

from src.helper import read_csv, load_config

logger = logging.getLogger(__name__)
project_path = path.dirname(path.dirname(path.abspath(__file__)))


def main(args):
    """
    main function to fune tuning bert classification model
    :param args: (argparse) user-input configuration file
    """
    try:
        config_path = project_path + "/" + args.config
        input_data_path = project_path + "/" + args.input
        model_path = project_path + "/" + args.model
        evaluation_path = project_path + "/" + args.evaluation

        config = load_config(config_path)

        # load data
        df = read_csv(input_data_path)
        # # -- debug
        # df = df[:100]
        # Encode the classes for BERT.
        encoder = preprocessing.LabelEncoder()
        df['label'] = encoder.fit_transform(df['label'])

        # Split data into training and test sets.
        X_train, X_test, y_train, y_test = training_test_split(df, **config['bert']['training_test_split'])

        # Bert tokenization
        logger.info("Tokenizing...")
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        if not args.max_length:
            max_length = config['bert']['max_length']
        else:
            max_length = int(args.max_length)

        # DataLoaders for running the model
        if not args.batch_size:
            batch_size = config['bert']['batch_size']
        else:
            batch_size = int(args.batch_size)

        dataloader_train = pro_pipline(X_train, tokenizer, max_length, config['bert']['tokenize'], batch_size, y_train)
        dataloader_test = pro_pipline(X_test, tokenizer, max_length, config['bert']['tokenize'], batch_size, y_test)

        # Initialize the model.
        model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                           num_labels=df['label'].nunique(),
                                                                           output_attentions=False,
                                                                           output_hidden_states=False)
        # Setting optimizer
        optimizer = AdamW(model.parameters(), **config['bert']['optimizer'])

        # Setting epochs
        if not args.num_epoch:
            epochs = config['bert']['num_epoch']
        else:
            epochs = int(args.num_epoch)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(dataloader_train) * epochs)

        # Setting seeds
        seed = config['bert']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Write prints to .txt
        model_name = 'max_length' + str(max_length) + 'batch_size' + str(batch_size) + 'num_epoch' + str(epochs)
        e_dir = evaluation_path + "/" + model_name
        if not os.path.exists(e_dir):
            os.makedirs(e_dir)
        sys.stdout = open(e_dir + "/" + model_name + '.txt', 'w')
        logger.info("Training... and evaluations will be saved into %s", e_dir)

        device = torch.device('cuda')
        # device = torch.device('cpu')
        model.to(device)

        complete_epoch, training_loss, test_accuracy = [], [], []

        for epoch in tqdm(range(1, epochs + 1)):
            model.train()
            loss_train_total = 0
            progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
            for batch in progress_bar:
                model.zero_grad()
                batch = tuple(b.to(device) for b in batch)
                inputs = {'input_ids': batch[0].to(device),
                          'attention_mask': batch[1].to(device),
                          'labels': batch[2].to(device),
                          }
                outputs = model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
            # training loss
            tqdm.write(f'\nEpoch {epoch}')
            loss_train_avg = loss_train_total / len(dataloader_train)
            training_loss.append(loss_train_avg)
            tqdm.write(f'Training loss: {loss_train_avg}')
            # evaluate the model
            plt, val_accuracy = run_evaluation(dataloader_test, model, device, encoder)
            plt.savefig(e_dir + "/" + model_name + '-' + str(epoch) + '.png')

            test_accuracy.append(val_accuracy)
            complete_epoch.append(epoch)
            loss_plt = plot_loss(complete_epoch, training_loss, test_accuracy)
            loss_plt.savefig(e_dir + "/" + model_name + '_loss' + '.png')

        # save the model for future use/retrain
        output_dir = model_path + '/' + model_name + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logging.info("Saving model to %s" % output_dir)

        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    except KeyError as e3:
        logger.error("KeyError: " + str(e3))
    except Exception as e:
        logger.error("Unexpected error occurred when training with Bert: " + str(e))


def training_test_split(df, random_state, test_size):
    # Set X and y
    X = df['line']
    y = df['label']

    # Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    return X_train, X_test, y_train, y_test


def pro_pipline(X, tokenizer, max_length, params, batch_size, y=None):
    encoded_data = tokenizer.batch_encode_plus(X, max_length=max_length, **params)
    input_ids_val = encoded_data['input_ids']
    attention_masks_val = encoded_data['attention_mask']
    if y is not None:
        labels_val = torch.tensor(y.values)
        dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    else:
        dataset_val = TensorDataset(input_ids_val, attention_masks_val)
    dataloader = DataLoader(dataset_val, sampler=RandomSampler(dataset_val), batch_size=batch_size)
    return dataloader


def run_evaluation(dataloader_test, model, device, encoder):
    """
    function to run the evaluation for BERT model
    :param dataloader_test: test set
    :param model: the BERT model object
    :param device: which device to use
    :param encoder: the encoder used to covert raw labels
    :param epoch: the current epoch of the training
    :return: None
    """
    # Validation Loss and Validation F-1 Score
    val_loss, predictions, true_vals = evaluate(dataloader_test, model, device)
    val_f1 = f1_score_func(predictions, true_vals)
    print('Test Loss = ', val_loss)
    print('Test F1 Score = ', val_f1)

    # Validation Accuracy
    encoded_classes = encoder.classes_
    predicted_category = [encoded_classes[np.argmax(x)] for x in predictions]
    true_category = [encoded_classes[x] for x in true_vals]

    # accuracy score
    val_accuracy = accuracy_score(true_category, predicted_category)
    print('Test Accuracy Score = ', val_accuracy)

    # Classification Report
    report = classification_report(true_category, predicted_category)
    print('Test Classification Report = ')
    print(report)

    # Confusion Matrix
    confusion = confusion_matrix(true_category, predicted_category)
    confusion_df = pd.DataFrame(confusion, index=list(encoder.classes_), columns=list(encoder.classes_))
    print(confusion_df)

    plt.clf()
    sns.heatmap(confusion_df / np.sum(confusion_df), annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    return plt, val_accuracy


def evaluate(dataloader_test, model, device):
    """
    function to evaluate the BERT model
    :param dataloader_test: test set
    :param model: the BERT model object
    :param device: the device that BERT model is working on
    :return: loss_val_avg, predictions, true_vals
    """
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in dataloader_test:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }
        with torch.no_grad():
            outputs = model(**inputs)
        # test loss
        loss = outputs[0]
        loss_val_total += loss.item()

        # Prediction
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

        # True labels
        label_ids = inputs['labels'].cpu().numpy()
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_test)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


def f1_score_func(preds, labels):
    """
    function to calculate F-1 score for the given predicted labels and true labels
    :param preds: predicted values from the BERT model
    :param labels: true labels
    :return: the corresponding F-1 score
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def plot_loss(epoch, training_loss, test_accuracy):
    plt.clf()
    plt.plot(epoch, training_loss, label='Training Loss')
    plt.plot(epoch, test_accuracy, label='Test Accuracy')

    plt.title('Avg Training Loss/Test Accuracy v.s. Epoch')
    plt.xlabel('Epoch')
    plt.legend()
    return plt
