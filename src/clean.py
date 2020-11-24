import logging
import pandas as pd
import re
from os import path
import sqlite3

from src.helper import save_csv, load_config

logger = logging.getLogger(__name__)
project_path = path.dirname(path.dirname(path.abspath(__file__)))


def main(args):
    """
    main function to load raw data, clean data and save leaned data to csv
    :param args: (argparse) user-input configuration file
    """
    try:
        config_path = project_path + "/" + args.config
        input_data_path = project_path + "/" + args.input
        output_data_path = project_path + "/" + args.output

        config = load_config(config_path)

        # load data
        logger.info("Trying to load data from %s", input_data_path)
        with open(input_data_path, 'r') as f:
            text = f.read()
        logger.info("Successfully loaded data from {}".format(input_data_path))

        clean_data = clean(text, **config['clean'])

        # Write to output file
        save_csv(clean_data, output_data_path)
    except KeyError as e3:
        logger.error("KeyError: " + str(e3))
    except FileNotFoundError as e1:
        logger.error('FileNotFoundError: {}'.format(e1))
    except Exception as e:
        logger.error("Unexpected error occurred when cleaning data: " + str(e))


def clean(text, labels, min_len, ignore_non_alphabetic):
    """
    Extract labels and texts form string, clean data
    Args:
        text: (String) a string
        labels: Characters we want to analysis
        ignore_non_alphabetic: True if ignore all non_alphabetic characters
        min_len: ignore lines with less than x tokens(Note: len(line) is effected by ignore_non_alphabetic)

    Returns: (DataFrame) DataFrame with three columns
            - 'charactor': Name of the charactors
            - 'line': lines
            - 'linelen': length of the line

    """

    # A Sample of a line: "Phoebe: Wait, does he eat chalk?\n"
    lines = text.split('\n')

    # extract episode # and season #
    episode_names = [i for i in lines if re.match(r"[0-9]{3,4}", i)]
    episodes = [re.findall(r"[0-9]{3,4}", i)[0] for i in episode_names]
    seasons = [i[:-2] for i in episodes]
    index = [lines.index(i) for i in episode_names]

    ind_df = pd.DataFrame(zip(episode_names, episodes, seasons, index),
                          columns=['episode_name', 'episode', 'season', 'idx'])
    ind_df['idx_shift'] = ind_df['idx'].shift(periods=-1).fillna(100000)
    ind_df['idx_shift'] = ind_df['idx_shift'] - 1

    # Append episode and season info to each line
    raw_df = pd.DataFrame(lines, columns=['raw_line'])
    raw_df['idx'] = raw_df.index

    conn = sqlite3.connect(':memory:')
    raw_df.to_sql('raw_df', conn, index=False)
    ind_df.to_sql('ind_df', conn, index=False)

    qry = '''
        select  
            raw_df.raw_line,
            ind_df.episode_name,
            ind_df.episode,
            ind_df.season
        from
            raw_df 
            join 
            ind_df 
            on
            raw_df.idx between ind_df.idx and ind_df.idx_shift
    '''

    df = pd.read_sql_query(qry, conn)

    # Extract labels
    df.loc[:, 'label'] = [i[0].lower() for i in df['raw_line'].str.split(":", 1)]
    df = df[df['label'].isin(labels)]

    # Extract lines
    df.loc[:, 'line'] = [i[1] for i in df['raw_line'].str.split(":", 1)]
    if ignore_non_alphabetic:
        df['line'] = df['line'].str.replace("[^a-zA-Z\s]+", "")

    df.loc[:, 'linelen'] = [len(i) for i in df['line'].str.split()]
    df = df[df['linelen'] >= min_len]

    return df
