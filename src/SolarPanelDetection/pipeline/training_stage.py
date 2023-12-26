from pathlib import Path
from SolarPanelDetection import logger
from SolarPanelDetection.utils.common import read_csv
from SolarPanelDetection.config.configuration import ConfigurationManager
from SolarPanelDetection.components.training import Training


STAGE_NAME = "Training Stage"

class TrainingPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.training_config = config.get_training_config()
        self.df = read_csv(Path(self.training_config.df_path))

    def main(self):
        training = Training(self.df, self.training_config)
        training.train()


if __name__=='__main__':
    try:
        logger.info(f"============= {STAGE_NAME} started =============")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f"============= {STAGE_NAME} completed =============")

    except Exception as e:
        logger.exception(e)
        raise e