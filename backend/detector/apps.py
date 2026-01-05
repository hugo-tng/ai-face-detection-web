from django.apps import AppConfig
import os
from django.conf import settings
from inference.predictor import DeepfakePredictor

class DetectorConfig(AppConfig):
    name = "detector"

    def ready(self):
        import os
        from django.conf import settings
        from inference.predictor import DeepfakePredictor

        model_dir = os.path.join(settings.BASE_DIR.parent, "models")
        self.predictor = DeepfakePredictor(model_dir)

