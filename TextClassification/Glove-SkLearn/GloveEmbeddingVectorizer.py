import sys
sys.path.append('../../')
from TextPreprocess import Preprocessing 
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
class AverageEmbeddingVectorizer(object):

    def __init__(self, embeddings = None):
        self.embeddings = embeddings
        self.dim = 50

    def __getstate_(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("transform is called\n")
        xTokens = [Preprocessing.processing(line, remove_stopwords = False) for line in X]
        
        return np.array([
                np.mean([self.embeddings[w]
                         for w in words if w in self.embeddings] or
                        [np.zeros(self.dim)], axis=0)
                for words in xTokens
            ])

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)