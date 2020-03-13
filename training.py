from pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    data_path = r'C:\Users\Achilles\PycharmProjects\data\Spotify'
    # data = DataParser(data_path).data
    random_state = 1337
    params = {'min_df': 0.01, 'max_df': 0.25, 'ngrams': (1, 3), 'clusters': 5}
    TrainingPipeline(data=None, params=params, random_state=random_state)
