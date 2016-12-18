import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
class Preprocessing(object):
    @staticmethod
    def processing(text):

        stop_words = stopwords.words('english')
        tokens = [word for word in nltk.word_tokenize(text.lower()) if word not in stop_words and len(word) >= 3]

        lemma = WordNetLemmatizer()
        tokens = [lemma.lemmatize(token) for token in tokens]

        return tokens
    


