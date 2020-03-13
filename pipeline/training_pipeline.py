import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import pickle
from time import time


class TrainingPipeline:
    def __init__(self, params=None, data=None, random_state=1337):
        self.X = None
        self.params = params
        self.random_state = random_state
        if data is None:
            self.data = pd.read_csv(os.getcwd() + r'\modelling_data.csv')
        else:
            self.data = data
        self.data = self.data.dropna(subset=['message_body'])
        self.X = self.data['message_body'].values
        self.pipeline = self.train_model()
        self.get_results()
        self.annotate_data()
        self.export_data()
        self.pickle_model()

    def train_model(self):
        print('Training...')
        t0 = time()
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=self.params['max_df'],
                                     min_df=self.params['min_df'],
                                     stop_words='english',
                                     ngram_range=self.params['ngrams'])
        pipeline = make_pipeline(vectorizer, TfidfTransformer(), Normalizer(copy=False),
                                 KMeans(n_clusters=self.params['clusters'], init='k-means++',
                                        max_iter=100, n_init=1, verbose=False, random_state=self.random_state))
        pipeline.fit(self.X)
        print("done in %0.3fs" % (time() - t0))
        return pipeline

    def get_results(self):
        res = self.pipeline.steps[3][1].cluster_centers_.argsort()[:, ::-1]
        terms = self.pipeline.steps[0][1].get_feature_names()
        for cluster in range(len(res)):
            print("> Cluster %d:" % cluster, end='')
            for ind in res[cluster, :20]:
                print(' %s' % terms[ind], end='')
            print('\n')

    def annotate_data(self):
        topic_map = {0: 'General', 1: 'Update related issue', 2: 'Content related query', 3: 'Technical issue',
                     4: 'Subscription and account related issue'}
        self.data['topic_number'] = self.pipeline.predict(self.data['message_body'].values)
        self.data['topic_name'] = self.data['topic_number'].map(topic_map)
        self.plot_data_distribution()

    def plot_data_distribution(self):
        self.data.topic_name.hist(bins=5)
        plt.xticks(rotation=90)
        plt.show()

    def export_data(self):
        self.data.to_csv('annotated_data.csv')

    def pickle_model(self):
        pickle.dump(self.pipeline, open(os.getcwd() + r'\model.pkl', 'wb'))
