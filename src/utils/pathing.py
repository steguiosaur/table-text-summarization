from pathlib import Path
import os


class Pathing:
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path("../../res")
    MODELS_PATH = OUTPUT_PATH / Path("../../models")

    @staticmethod
    def asset_path(path: str) -> Path:
        return Pathing.ASSETS_PATH / Path(path)

    @staticmethod
    def model_path(path: str) -> Path:
        return Pathing.MODELS_PATH / Path(path)
