import argparse

from configs.utils import get_config
from builders.trainer_builder import build_trainer
from utils.logging_utils import setup_logger

logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, default="configs/object_relation_transformer.yaml")

args = parser.parse_args()

config = get_config(args.config_file)

trainer = build_trainer(config)
trainer.start()
# trainer.get_predictions(get_scores=config.TRAINING.GET_SCORES)
logger.info("Trainer done.")