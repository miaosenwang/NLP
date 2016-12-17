from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

class TrainClassifier(object):
    @staticmethod
    def Run(trainData, label, filename, vectorizer = TfidfVectorizer()):

        print("Start Training...")

        X_train = vectorizer.fit_transform(trainData)
        bdt = AdaBoostClassifier(DecisionTreeClassifier())
        bdt.fit(X_train, label)

        print("Done Training")

        print("Start Saving...")
        joblib.dump(bdt, filename)     
        print("Done Saving")
        return bdt