import csv

class DataSet(object):

    def load(self, path):
        self.Label = []
        self.Data = []

        with open(path,'r', encoding="utf8") as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                self.Label.append(int(row[0]))
                self.Data.append(row[1])


