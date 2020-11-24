import traceback
from flask import render_template, request, redirect, url_for
import logging.config
from flask import Flask
from os import path
import transformers
import torch
from src.helper import load_config
import config.config as config
import torch.nn.functional as F
import numpy as np
from src.bert_classification import pro_pipline
from sklearn import preprocessing
import argparse


logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__, template_folder="templates")

# Configuration File
config_path = config.CONFIG_YAML
configs = load_config(config_path)

# Load model
project_path = path.dirname(path.abspath(__file__))

# which model to load
parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", help="number of epochs for training the model")
parser.add_argument("--batch_size", help="batch_size for training the model")
parser.add_argument("--max_length", help="max length of reviews")
args = parser.parse_args()

if not args.max_length:
    max_length = configs['bert']['max_length']
else:
    max_length = int(args.max_length)

# DataLoaders for running the model
if not args.batch_size:
    batch_size = configs['bert']['batch_size']
else:
    batch_size = int(args.batch_size)
if not args.num_epoch:
    num_epoch = configs['bert']['num_epoch']
else:
    num_epoch = int(args.num_epoch)

model_name = 'max_length' + str(max_length) + 'batch_size' + str(batch_size) + 'num_epoch' + str(num_epoch)

model_path = project_path + "/" + config.model_path + '/' + model_name
model = transformers.BertForSequenceClassification.from_pretrained(model_path)
# device = torch.device('cuda')
device = torch.device('cpu')

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


encoder = preprocessing.LabelEncoder()
encoder.fit(configs['clean']['labels'])


def bert_predict(dataloader, model, device, encoder):
    """
    function to evaluate the BERT model
    :param dataloader_test: test set
    :param model: the BERT model object
    :param device: the device that BERT model is working on
    :return: loss_val_avg, predictions, true_vals
    """
    model.eval()
    for batch in dataloader:
        b_input_ids, b_attn_mask = tuple(b.to(device) for b in batch)[:2]
        # Prediction
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

    logits = F.softmax(logits[0], dim=1).cpu().numpy()
    encoded_classes = encoder.classes_
    predicted_category = [encoded_classes[np.argmax(x)] for x in logits]

    return predicted_category, max(logits[0])


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_dialogue = request.form["dialogue"]
        dataloader = pro_pipline([input_dialogue], tokenizer, max_length, configs['bert']['tokenize'], batch_size)
        predicted_category, prob = bert_predict(dataloader, model, device, encoder)

        return render_template('main.html',
                               predicted_category=predicted_category[0].capitalize(),
                               dialogue=input_dialogue,
                               prob=str(round(prob * 100, 2)) + "%")

    except:
        traceback.print_exc()
        logger.warning("Not able to get recommendations, error page returned")
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)
