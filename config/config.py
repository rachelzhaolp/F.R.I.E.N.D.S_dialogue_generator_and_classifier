from os import path
PROJECT_HOME = path.dirname(path.abspath(__file__))


"""
config for default path arguments in run.py
"""
# config
CONFIG_YAML = "config/config.yaml"

# raw data
RAW_DATA = "data/friends_dialogue.txt"

# clean data
CLEAN_DATA = "data/cleaned_dialogue.csv"

# eda data
EDA_DATA = "EDA/eda.txt"

# org + aug data
AUG_DATA = "data/augmented_dialogue.csv"

# model directory
model_path = "models/"

# evaluation directory
evaluation_path = "evaluations/"

# org + aug data
GPT_GENERATED = "data/gpt_new_dialogue.csv"
