import timm
from .resnet import resnet18, resnet50, resnet101

class ModelFactory:
    _model_zoo = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "vit": "vit_tiny_patch16_224",  # Vision Transformer Tiny
        "vits": "vit_small_patch16_224",
        "vitb": "vit_base_patch16_224",
    }

    @classmethod
    def create(cls, model: str, args, num_classes: int, **kwargs):
        if model not in cls._model_zoo:
            raise ValueError(f"Model {model} not found in our model zoo.")

        if model.startswith("vit"):
            # For Vision Transformers, use the timm library.
            model_name = cls._model_zoo[model]
            model_instance = timm.create_model(model_name, pretrained=False, num_classes=num_classes, **kwargs)
        else:
            # Use the function from the model zoo to create an instance.
            model_instance = cls._model_zoo[model](args, num_classes=num_classes, **kwargs)

        return model_instance