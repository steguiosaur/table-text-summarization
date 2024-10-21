from pathlib import Path


class Pathing:
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path("../../res")

    @staticmethod
    def asset_path(path: str) -> Path:
        return Pathing.ASSETS_PATH / Path(path)
