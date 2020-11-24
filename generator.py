import argparse
import logging
import config.config as config
from src.gpt_generator import main as generate
from src.finetuning import main as finetune

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Add parser for the model pipeline
    parser = argparse.ArgumentParser(description="Classification Model pipeline")
    subparsers = parser.add_subparsers()

    # finetune parser
    sb_ft = subparsers.add_parser("finetune",
                                  description="Fine-tuning GPT-2 with .txt")
    sb_ft.add_argument('--config', default=config.CONFIG_YAML, help='Path to yaml configuration file')
    sb_ft.add_argument('--input', '-i', default=config.RAW_DATA, help='Path to input data')
    sb_ft.set_defaults(func=finetune)

    # generator parser
    sb_ft = subparsers.add_parser("generate_sample",
                                  description="Generating sample data with finetuned model")
    sb_ft.add_argument('--config', default=config.CONFIG_YAML, help='Path to yaml configuration file')
    sb_ft.add_argument('--input', '-i', default=config.CLEAN_DATA, help='Path to input data')
    sb_ft.add_argument('--output', '-o', default=config.GPT_GENERATED, help='Path to save output (optional)')
    sb_ft.set_defaults(func=generate)

    args = parser.parse_args()
    args.func(args)
