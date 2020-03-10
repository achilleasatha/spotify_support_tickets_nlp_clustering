import pandas as pd
import os
from data_parser.data_parser import DataParser
from pipeline.training_pipeline import TrainingPipeline
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    data_path = r'C:\Users\Achilles\PycharmProjects\data\Spotify'
    data = pd.read_csv(data_path + r'\spotify-public-dataset.csv')

    random_state = 1337

    model = RandomForestClassifier()

    param_grid = {'hyperparams': []}

    TrainingPipeline(model, data, data_path, param_grid, random_state)