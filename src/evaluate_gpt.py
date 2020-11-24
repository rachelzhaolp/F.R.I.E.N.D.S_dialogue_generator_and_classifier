import logging
import pandas as pd
from os import path
import itertools
import torch
import transformers
from sklearn import preprocessing
from src.helper import read_csv, load_config
from src.bert_classification import run_evaluation, pro_pipline
import sys
import os

logger = logging.getLogger(__name__)
project_path = path.dirname(path.dirname(path.abspath(__file__)))


def main(args):
    """
    main function to clean data
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

        logging.info("Prepossessing data...")
        df_new = clean(df, **config['clean'])
        # # -- debug
        # df_new = df_new[:100]

        # processing data
        # encoding y
        encoder = preprocessing.LabelEncoder()
        df_new['label'] = encoder.fit_transform(df_new['label'])

        # enbedding X
        X = df_new['line']
        y = df_new['label']
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        if not args.max_length:
            max_length = config['bert']['max_length']
        else:
            max_length = int(args.max_length)

        if not args.batch_size:
            batch_size = config['bert']['batch_size']
        else:
            batch_size = int(args.batch_size)

        dataloader_val = pro_pipline(X, tokenizer, max_length, config['bert']['tokenize'], batch_size, y)

        logging.info("Loading model...")
        # load model
        if not args.num_epoch:
            epochs = config['bert']['num_epoch']
        else:
            epochs = int(args.num_epoch)

        model_name = 'max_length' + str(max_length) + 'batch_size' + str(batch_size) + 'num_epoch' + str(epochs)
        model_dir = model_path + '/' + model_name + "/"
        model = transformers.BertForSequenceClassification.from_pretrained(model_dir)
        # device = torch.device('cuda')
        device = torch.device('cpu')
        model.to(device)

        e_dir = evaluation_path + "/gpt/"
        if not os.path.exists(e_dir):
            os.makedirs(e_dir)

        logging.info("Runing... Evaluation criterias will be saved in {}.".format(e_dir))
        sys.stdout = open(e_dir + "/" + model_name + '.txt', 'w')
        plt, val_accuracy = run_evaluation(dataloader_val, model, device, encoder)
        plt.savefig(e_dir + "/" + model_name + '.png')

    except Exception as e:
        logger.error("Unexpected error occurred when evaluation: " + str(e))


def clean(df, labels, min_len, ignore_non_alphabetic):
    new = [i[1:] for i in list(df['raw_line'].str.split("\n"))]
    merged = list(itertools.chain.from_iterable(new))

    sep = [i.split(":", 1) for i in merged]
    sep = [i for i in sep if (i[0].lower() in labels) and (len(i) == 2)]

    df_new = pd.DataFrame(sep, columns=['label', 'line'])
    df_new['label'] = df_new['label'].str.lower()

    # Extract lines
    if ignore_non_alphabetic:
        df_new['line'] = df_new['line'].str.replace("[^a-zA-Z\s]+", "")

    df_new.loc[:, 'linelen'] = [len(i) for i in df_new['line'].str.split()]
    df_new = df_new[df_new['linelen'] >= min_len]

    return df_new
