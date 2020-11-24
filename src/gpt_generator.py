import logging
from os import path
import pandas as pd
import random
import gpt_2_simple as gpt2

from src.helper import read_csv, load_config, save_csv

logger = logging.getLogger(__name__)
project_path = path.dirname(path.dirname(path.abspath(__file__)))


def main(args):
    """
    Load generated model checkpoints from by default in /checkpoint/run1 and generate new text
    """
    try:
        config_path = project_path + "/" + args.config
        input_data_path = project_path + "/" + args.input
        output_data_path = project_path + "/" + args.output

        config = load_config(config_path)

        # load data
        df = read_csv(input_data_path)
        lines = list(df['raw_line'])
        random.seed(config['generate']['random_seed'])
        sample_seeds = random.choices(lines, k=config['generate']['num'])

        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess)

        pred = []
        for i in sample_seeds:
            out = gpt2.generate(sess, prefix=i, **config['generate']['generator'])
            pred.append(out)

        pred_df = pd.DataFrame(pred, columns=['raw_line'])
        save_csv(pred_df, output_data_path)

    except Exception as e:
        logger.error("Unexpected error occurred when generating dialogues with gpt2: " + str(e))
