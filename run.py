import argparse
import logging
import config.config as config
from src.clean import main as clean
from src.eda import main as eda
from src.augmentation import main as augment
from src.bert_classification import main as bert
from src.evaluate_gpt import main as gpt

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Add parser for the model pipeline
    parser = argparse.ArgumentParser(description="Classification Model pipeline")
    subparsers = parser.add_subparsers()

    # Clean parser
    sb_clean = subparsers.add_parser("clean_data",
                                     description="Load data from data_source and save cleaned data to out_filepath")
    sb_clean.add_argument('--config', default=config.CONFIG_YAML, help='Path to yaml configuration file')
    sb_clean.add_argument('--input', '-i', default=config.RAW_DATA, help='Path to input data')
    sb_clean.add_argument('--output', '-o', default=config.CLEAN_DATA, help='Path to save output (optional)')
    sb_clean.set_defaults(func=clean)

    # Eda parser
    sb_eda = subparsers.add_parser("eda",
                                   description="Exploratory Analysis on cleaned data(output of clean.py)")
    sb_eda.add_argument('--config', default=config.CONFIG_YAML, help='Path to yaml configuration file')
    sb_eda.add_argument('--input', '-i', default=config.CLEAN_DATA, help='Path to input data')
    sb_eda.add_argument('--output', '-o', default=config.EDA_DATA, help='Path to save output (optional)')
    sb_eda.set_defaults(func=eda)

    # Contextual WordEmbs Augmentation parser
    sb_aug = subparsers.add_parser("augment",
                                   description="Load cleaned data(output of clean.py), perform data augmentation and "
                                               "save all samples to out_filepath")
    sb_aug.add_argument('--config', default=config.CONFIG_YAML, help='Path to yaml configuration file')
    sb_aug.add_argument('--input', '-i', default=config.CLEAN_DATA, help='Path to input data')
    sb_aug.add_argument('--output', '-o', default=config.AUG_DATA, help='Path to save output (optional)')
    sb_aug.set_defaults(func=augment)

    # Bert classification parser
    sb_bert = subparsers.add_parser("bert",
                                    description="Load augmented data(output of augmentation.py) and fine tuning "
                                                "Bert(bert-base-uncased) for multiclass classification")
    sb_bert.add_argument('--config', default=config.CONFIG_YAML, help='Path to yaml configuration file')
    sb_bert.add_argument('--input', '-i', default=config.AUG_DATA, help='Path to input data')
    sb_bert.add_argument('--model', '-m', default=config.model_path, help='Path to save model')
    sb_bert.add_argument('--evaluation', '-e', default=config.evaluation_path, help='Path to save evaluations')
    sb_bert.add_argument("--num_epoch", help="number of epochs for training the model")
    sb_bert.add_argument("--batch_size", help="batch_size for training the model")
    sb_bert.add_argument("--max_length", help="max length of each sample")
    sb_bert.set_defaults(func=bert)

    # GPT evaluation  parser
    sb_gpt = subparsers.add_parser("gpt",
                                    description="Calculating Accuracy of dialogues generated for gpt2")
    sb_gpt.add_argument('--config', default=config.CONFIG_YAML, help='Path to yaml configuration file')
    sb_gpt.add_argument('--input', '-i', default=config.GPT_GENERATED, help='Path to input data')
    sb_gpt.add_argument('--model', '-m', default=config.model_path, help='Path to save model')
    sb_gpt.add_argument('--evaluation', '-e', default=config.evaluation_path, help='Path to save evaluations')
    sb_gpt.add_argument("--num_epoch", help="number of epochs for training the model")
    sb_gpt.add_argument("--batch_size", help="batch_size for training the model")
    sb_gpt.add_argument("--max_length", help="max length of each sample")
    sb_gpt.set_defaults(func=gpt)

    args = parser.parse_args()
    args.func(args)
