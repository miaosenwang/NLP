from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

class CrossValidation(object):
    @staticmethod
    def Run(data, label, vectorizer = TfidfVectorizer()):

        pipeline = Pipeline([
            ('vect', vectorizer),
            ('adaTree', AdaBoostClassifier(DecisionTreeClassifier()))
            ])

        parameters = {
            'vect__max_df':(0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000, 50000),
            'vect__ngram_range': ((1,1), (1,2), (1,3))
            }
        
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)

        grid_search.fit(data, label)

        print("best score: %.3f"% grid_search.best_score_)

        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()

        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        return best_parameters


 

