import numpy as np
class GloveHelper(object):
    @staticmethod
    def loadGlove(path):
        embeddings_index = {}
        with open(path, "r", encoding="utf8") as glovef:
            for line in glovef:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index



