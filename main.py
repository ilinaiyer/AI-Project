import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers
import albumentations as A

import kagglehub
from pathlib import Path

def read_link (link):
    return 'https://drive.google.com/uc?export=download&id='+link.split('/')[-2]
riasec_df = pd.read_csv(read_link('https://drive.google.com/file/d/1Yw8Q-okC156xESWz9ZdYJY8kOZpl3SjR/view?usp=sharing'))
riasec_df = pd.concat([riasec_df, # Realistic
                      pd.read_csv(read_link('https://drive.google.com/file/d/1fTj0tFJtQ4htBEa1dpLVjhWe93PYOHA1/view?usp=sharing')), # Investigative
                      pd.read_csv(read_link('https://drive.google.com/file/d/1CR2IHnrhKC-7EtUjP5s-x8nkie6zSfiZ/view?usp=sharing')), # Artistic
                      pd.read_csv(read_link('https://drive.google.com/file/d/1Me5geVIjvEtMPldfzMOwBWavJDjy-e0c/view?usp=sharing')), # Social
                      pd.read_csv(read_link('https://drive.google.com/file/d/1WOTNJ7htmu5jR3gvaCPNLKg0ca1p3jaR/view?usp=sharing')), # Enterprising
                      pd.read_csv(read_link('https://drive.google.com/file/d/1crJJh-svX5jGfVgs3oZlciEWJas_YYPY/view?usp=sharing')) # Conventional
                      ])
