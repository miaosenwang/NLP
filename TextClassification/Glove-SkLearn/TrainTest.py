from GloveHelper import GloveHelper
from TrainClassifier import TrainClassifier
from DataSet import DataSet
from TfidfEmbeddingVectorizer import TfidfEmbeddingVectorizer
from Predict import Predict

if __name__ == "__main__":
    embeddings = GloveHelper.loadGlove("C:\\Code\\Data\\Glove\\glove.6B\\glove.6B.Sample.txt")
    trainData = DataSet()
    trainData.load("C:\Code\Data\MeetingSummary\Ep\R1Train.txt")

    vectorizer = TfidfEmbeddingVectorizer(embeddings)
    model = TrainClassifier.Run(trainData.Data, trainData.Label, "model.pkl", vectorizer)

    testData = DataSet()
    testData.load("C:\Code\Data\MeetingSummary\Ep\R1Test.txt")

    predictor = Predict(vectorizer, model)
    result = predictor.Run(testData.Data, "C:\Code\Data\MeetingSummary\Ep\PyTestOut.txt")



