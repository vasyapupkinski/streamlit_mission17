from transformers import pipeline
import streamlit as st
from PIL import Image

class ImageClassificationService:
    def __init__(self):
        self.classifier = self._load_pipeline()
    
    @staticmethod
    
    @st.cache_resource
    def _load_pipeline():
        return pipeline(
            "image-classification",
            model="google/vit-base-patch16-224"
        )

    def predict(self, image: Image.Image):
        return self.classifier(image)