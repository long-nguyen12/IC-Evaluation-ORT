from .registry import Registry

META_TRAINER = Registry("TRAINER")

def build_trainer(config):
    # print(config)
    trainer = META_TRAINER.get(config.TRAINER)(config)

    return trainer