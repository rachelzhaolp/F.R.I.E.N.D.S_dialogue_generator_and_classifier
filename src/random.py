from os import path
import yaml
import os

project_path = path.dirname(path.dirname(path.abspath(__file__)))


def load_config(config_path):
    """
    Load config.yaml
    Args:
        config_path: (String)

    Returns:

    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(args):
    if not args.random:
        a = 100
    else:
        a = args.random
    print(type(a))
