
from .embedder import EmbeddingModel, ALL_EMBEDDERS
#from .classifier import ImageClassifier, ALL_CLASSIFIERS
from .vlm import VLM, ALL_VLMS
from .image_generator import ImageGenerator, ALL_IMG_GENS

def get_model(model_name, device):
    if model_name in ALL_EMBEDDERS:
        return EmbeddingModel(model_name, device)
    #if model_name in ALL_CLASSIFIERS:
    #    return ImageClassifier(model_name, device)
    if model_name in ALL_VLMS:
        return VLM(model_name, device)
    if model_name in ALL_IMG_GENS:
        return ImageGenerator(model_name, device)
