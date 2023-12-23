from SolarPanelDetection import logger
from SolarPanelDetection.pipeline.training_stage import TrainingPipeline
from SolarPanelDetection.pipeline.data_ingestion_stage import DataIngestionPipeline
from SolarPanelDetection.pipeline.data_preparation_stage import DataPreparationPipeline


STAGE_NAME = 'Data Ingestion Stage'
try:
    logger.info(f"============= {STAGE_NAME} started =============")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f"============= {STAGE_NAME} completed =============")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Preparation Stage"
try:
    logger.info(f"============= {STAGE_NAME} started =============")
    obj = DataPreparationPipeline()
    obj.main()
    logger.info(f"============= {STAGE_NAME} completed =============")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training Stage"
try:
    logger.info(f"============= {STAGE_NAME} started =============")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f"============= {STAGE_NAME} completed =============")

except Exception as e:
    logger.exception(e)
    raise e