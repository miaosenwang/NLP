from sklearn.feature_extraction.text import TfidfVectorizer

class Predict(object):
    def __init__(self, vectorizer = TfidfVectorizer(), model = None):
        self.vectorizer = vectorizer
        self.model = model

    def Run(self, testData, outputPath):    
        X_test = self.vectorizer.transform(testData)
        result = self.model.predict(X_test)

        if(outputPath == None):
            for label in result:
                print(label[0].astype(str) + '\n')
        else:
            with open(outputPath,'w', encoding="utf8") as f:
                for label in result:
                    f.write(label.astype(str) + '\n')

        print("Done writting test results.")

