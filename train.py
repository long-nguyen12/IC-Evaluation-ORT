import argparse

from configs.utils import get_config
from builders.trainer_builder import build_trainer
from utils.logging_utils import setup_logger

logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, default="configs/object_relation_transformer_custom-uit.yaml")
parser.add_argument("--testing", type=bool, default=False)

args = parser.parse_args()

if __name__ == "__main__":
    config = get_config(args.config_file)
    trainer = build_trainer(config)
    if not args.testing:
        trainer.start()
    else:
        trainer.get_predictions(get_scores=config.TRAINING.GET_SCORES)
    logger.info("Trainer done.")