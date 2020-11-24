import logging
import sys
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from src.helper import read_csv, load_config

logger = logging.getLogger(__name__)
project_path = path.dirname(path.dirname(path.abspath(__file__)))


def main(args):
    """
    main function to load cleaned data, conduct eda, visualize most important tokens with tfidf score
    :param args: (argparse) user-input configuration file
    """
    try:
        config_path = project_path + "/" + args.config
        input_data_path = project_path + "/" + args.input
        output_data_path = project_path + "/" + args.output

        config = load_config(config_path)

        # load data
        df = read_csv(input_data_path)
        df.loc[:, 'season'] = df['season'].astype('int')

        sys.stdout = open(output_data_path, 'w')
        check_balance(df)
        check_linelen(df, config['eda']['quantile'])

        groups = config['eda']['groups']

        for i in range(len(groups)):
            df_top_words = most_important_words(df, groups[i], **config['eda']['top_n_words'])
            fig = plot_tfidf_classfeats_h(df_top_words)
            fig.savefig('{}/EDA/top_words_{}.png'.format(project_path, i))


    except Exception as e:
        logger.error("Unexpected error occurred when eda: " + str(e))


def check_balance(df):
    """
    Check class balance
    Args:
        df: (DataFrame)
    """
    logger.info("Printing class balance")
    print("Class Balance:\n")
    print(df.groupby('label').count()['line'] / len(df))
    print("------------------------------------------------")


def check_linelen(df, quantile):
    """
    Stats of the length of the lines
    Args:
        df: (DataFrame)
    """
    logger.info("Printing quntiles of line length")
    print("Quantile of line length:\n")
    print(df['linelen'].quantile(quantile))
    print("------------------------------------------------\n")
    print('Min length: ' + str(df['linelen'].min()) + "\n")
    print('Median length: ' + str(df['linelen'].mean()) + "\n")
    print('Mean length: ' + str(df['linelen'].median()) + "\n")
    print('Max length: ' + str(df['linelen'].max()) + "\n")


def most_important_words(df, groups, tfidfParams, top_n):
    logger.info("Extracting most important tokens with tf-idf")
    # generate documents for each group
    temp = df.groupby(groups)['line'].apply(lambda x: ','.join(x)).reset_index()
    temp = temp.sort_values(groups)
    temp = temp.astype(str)
    temp.loc[:, 'group'] = temp[groups].agg('-'.join, axis=1)
    # TF-IDF transformation
    vectorizer = TfidfVectorizer(**tfidfParams)
    tfidf = vectorizer.fit_transform(temp['line'])
    features = vectorizer.get_feature_names()
    # Extract top x for each group
    tops = tfidf.toarray().argsort(axis=1)[:, -top_n:]
    words = np.array(features)[tops]
    scores = [tfidf.toarray()[i][tops[i]] for i in range(tfidf.shape[0])]
    matching = [list(zip([i[0]] * top_n, [i[1]] * top_n, i[2], i[3])) for i in
                list(zip(temp['label'], temp['group'], words, scores))]
    df_top_words = pd.DataFrame([item for charactor in matching for item in charactor],
                                columns=['label', 'group', 'word', 'tfidf_score'])
    return df_top_words


def plot_tfidf_classfeats_h(df):
    logger.info("Visualizing tf-idf scores of the most important tokens")
    n_col = df['label'].nunique()
    n_row = int(df['group'].nunique() / df['label'].nunique())
    groups = df['group'].unique()

    fig1, axs = plt.subplots(n_row, n_col, figsize=(3 * n_col, 2 * n_row))
    fig1.subplots_adjust(left=0.2, wspace=0.6)

    for i in range(len(groups)):
        temp = df[df['group'] == groups[i]].sort_values('tfidf_score', ascending=False)
        y_pos = np.arange(len(temp))

        if n_row == 1:
            ax = axs[i]
            axs[0].set_ylabel('Full Seasons', rotation=90, size='large', fontsize=15)
            ax.set_title(groups[i], fontsize=15)
        else:
            ax = axs[i // n_col, i % n_col]

        ax.barh(y_pos, temp['tfidf_score'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(temp['word'])
        ax.invert_yaxis()  # labels read top-to-bottom

    if n_row > 1:
        cols = df['label'].unique()
        for ax, col in zip(axs[0], cols):
            ax.set_title(col, fontsize=15)

        rows = ["Season " + str(i + 1) for i in range(10)]
        for ax, row in zip(axs[:, 0], rows):
            ax.set_ylabel(row, rotation=90, size='large', fontsize=15)

        fig1.align_ylabels(axs[:, 0])

    plt.tight_layout()
    return fig1
