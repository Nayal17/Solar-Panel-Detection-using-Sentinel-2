from SolarPanelDetection import logger
from SolarPanelDetection.config.configuration import ConfigurationManager
from SolarPanelDetection.components.prepare_data import DataPreparation


STAGE_NAME = "Data Preparation Stage"

class DataPreparationPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_preparation_config = config.get_data_preparation_config()

    def main(self):
        data_preparation = DataPreparation(self.data_preparation_config)
        data_preparation.get_features()


if __name__=='__main__':
    try:
        logger.info(f"============= {STAGE_NAME} started =============")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f"============= {STAGE_NAME} completed =============")

    except Exception as e:
        logger.exception(e)
        raise e