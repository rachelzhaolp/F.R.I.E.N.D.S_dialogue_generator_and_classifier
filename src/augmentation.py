import logging
from os import path
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import multiprocessing as mp
from nlpaug.augmenter.word import ContextualWordEmbsAug
from nlpaug.util import Action
import pandas as pd

from src.helper import read_csv, save_csv, load_config

logger = logging.getLogger(__name__)
project_path = path.dirname(path.dirname(path.abspath(__file__)))


def main(args):
    """
    main function perform data augmentation with clean data and save the augmented data to csv
    :param args: (argparse) user-input configuration file
    """
    # try:
    config_path = project_path + "/" + args.config
    input_data_path = project_path + "/" + args.input
    output_data_path = project_path + "/" + args.output

    config = load_config(config_path)

    # load data
    df = read_csv(input_data_path)

    lines = list(df['line'])
    charactors = list(df['label'])

    augmented = augment(lines, config['aug'])

    # Union original lines and augmented lines
    df2 = pd.DataFrame(list(zip(charactors, augmented)),
                       columns=['label', 'line'])
    df = df[['label', 'line']]

    df['type'] = 'original'
    df2['type'] = 'augmented'
    result = pd.concat([df, df2])

    save_csv(result, output_data_path)

    # except Exception as e:
    #     logger.error("Unexpected error occurred when eda: " + str(e))


def augment(lines, params):
    """
    Contextual WordEmbs Augmentation with nlpaug for a list of String
    Args:
        lines: (List of Strings)
        params: (Dictionary) aug_max arguments

    Returns: (List of Strings) new strings

    """
    # Contextual WordEmbs Augumentation pipline
    aug = naf.Sequential(
        [
            ContextualWordEmbsAug(action=Action.INSERT, aug_max=params['contextual_max']),
            ContextualWordEmbsAug(action=Action.SUBSTITUTE, aug_max=params['contextual_max']),
            naw.RandomWordAug(aug_max=params['ramdom_max'])
        ]
    )

    augmented = []
    total_batchs = len(lines) // 100 + 1

    for i in range(total_batchs):
        if i % 100 == 0:
            logger.info("Augmenting the {} th batch, {}%".format(i, round(i / total_batchs * 100)))
        if i == total_batchs - 1:
            sub_line = lines[100 * i:]
        else:
            sub_line = lines[100 * i: 100 * (i + 1)]
        sub_aug = aug.augment(sub_line, num_thread=mp.cpu_count() - 1)
        augmented = augmented + sub_aug
    return augmented
