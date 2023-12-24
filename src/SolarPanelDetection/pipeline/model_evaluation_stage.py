from pathlib import Path
from SolarPanelDetection import logger
from SolarPanelDetection.utils.common import read_csv
from SolarPanelDetection.config.configuration import ConfigurationManager
from SolarPanelDetection.components.model_evaluation_mlflow import Evaluation


STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.evaluation_config = config.get_evaluation_config()
        self.df = read_csv(Path(self.evaluation_config.df_path))

    def main(self):
        evaluation = Evaluation(self.df, self.evaluation_config)
        evaluation.evaluate()
        evaluation.log_into_mlflow()


if __name__=='__main__':
    try:
        logger.info(f"============= {STAGE_NAME} started =============")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"============= {STAGE_NAME} completed =============")

    except Exception as e:
        logger.exception(e)
        raise e