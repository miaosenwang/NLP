import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
class Preprocessing(object):
    @staticmethod
    def processing(text, remove_stopwords):
        
        tokens = [word for word in nltk.word_tokenize(text.lower())]

        if(remove_stopwords):
            stop_words = stopwords.words('english')
            tokens = [word for word in tokens if word not in stop_words] 

        lemma = WordNetLemmatizer()
        result = [lemma.lemmatize(token) for token in tokens]

        return result
    


