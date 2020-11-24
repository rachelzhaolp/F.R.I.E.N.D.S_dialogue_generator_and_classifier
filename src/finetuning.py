import logging
from os import path
import os
import gpt_2_simple as gpt2

from src.helper import load_config

logger = logging.getLogger(__name__)
project_path = path.dirname(path.dirname(path.abspath(__file__)))


def main(args):
    """
    Fine tuning with pre-trained gpt2 model,
    The generated model checkpoints are by default in /checkpoint/run1.
    :param args: (argparse) user-input configuration file
    """
    try:
        config_path = project_path + "/" + args.config
        input_data_path = project_path + "/" + args.input

        config = load_config(config_path)

        # Download pre-trained model
        model_name = config['gpt2']['model_name']
        if not os.path.isdir(os.path.join("models", model_name)):
            logging.info(f"Downloading {model_name} model...")
            gpt2.download_gpt2(model_name=model_name)

        sess = gpt2.start_tf_sess()
        gpt2.finetune(sess,
                      input_data_path,
                      **config['gpt2'])  # steps is max number of training steps

    except Exception as e:
        logger.error("Unexpected error occurred when eda: " + str(e))

