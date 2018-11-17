from generate_feature import Features
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from passive_tagger import Tagger
import sys
import os
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pickle
np.random.seed(11)

class NaiveBayes:
    def __init__(self, features={}, split=0.8, distribution="Bernoulli", isSummary=False):        
        self.Tags = ["OTH", "BKG", "CTR", "NA", "AIM", "OWN", "BAS", "TXT", "", "BEGIN"]
        self.Locations = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        self.ParaLocations = ["INITIAL", "MEDIAL", "FINAL"]        
        self.Headlines = [
            "Introduction", "Implementation", "Example", "Conclusion", 
            "Result", "Evaluation", "Solution", "Discussion", 
            "Further Work", "Data", "Related Work", "Experiment", 
            "Problems", "Method", "Problem Statement", "Non-Prototypical"
        ]
        self.YESorNO = ["YES", "NO"]
        self.SecLocations = ["FIRST", "SECOND", "THIRD", "LAST",
            "SECOND-LAST", "THIRD-LAST", "SOMEWHERE"
        ]
        self.Tenses = ["PRESENT", "PAST", "FUTURE", "NOVERB"]
        self.Modals = ["MODAL", "NOMODAL", "NOVERB"]
        self.Voices = ["Active", "Passive", "NOVERB"]
        self.isSummary = isSummary
        self.features = features
        self.transformFeatures()
        self.distribution = distribution
        self.split = split
        self.splitData()
    
    def reloadDis(self):
        if self.distribution == "Bernoulli":
            self.nb = BernoulliNB()
        elif self.distribution == "Multinomial":
            self.nb = MultinomialNB()
        elif self.distribution == "Complement":
            self.nb = ComplementNB()
        else:
            self.nb = GaussianNB()
        
    def splitData(self):
        if not self.isSummary:
            print "Data split between train and test: " + str(self.split)
        papers = self.features.keys()
        order = np.random.permutation(len(papers))

        self.train_papers = []
        for i in range(int(self.split*len(papers))):
            self.train_papers.append(papers[order[i]])

        self.test_papers = []
        for i in range(int(self.split*len(papers))+1, len(papers)):
            self.test_papers.append(papers[order[i]])

        self.train_X, self.train_y = self.getFeatures(self.train_papers)
        self.test_X, self.test_y = self.getFeatures(self.test_papers)

    def transformFeatures(self):
        self.transformed_features = dict()
        for filename in self.features.keys():
            self.transformed_features[filename] = dict()
            for sentId in self.features[filename].keys():
                self.transformed_features[filename][sentId] = dict()
                self.transformed_features[filename][sentId]['loc'] = self.Locations.index(self.features[filename][sentId]['loc'])
                self.transformed_features[filename][sentId]['parloc'] = self.ParaLocations.index(self.features[filename][sentId]['parloc'])
                self.transformed_features[filename][sentId]['val'] = self.Tags.index(self.features[filename][sentId]['val'])
                self.transformed_features[filename][sentId]['Title'] = self.YESorNO.index(self.features[filename][sentId]['Title'])
                self.transformed_features[filename][sentId]['len'] = self.YESorNO.index(self.features[filename][sentId]['len'])
                self.transformed_features[filename][sentId]['tfidf'] = self.YESorNO.index(self.features[filename][sentId]['tfidf'])
                self.transformed_features[filename][sentId]['secloc'] = self.SecLocations.index(self.features[filename][sentId]['secloc'])
                self.transformed_features[filename][sentId]['Headlines'] = self.Headlines.index(self.features[filename][sentId]['Headlines'])
                self.transformed_features[filename][sentId]['history'] = self.Tags.index(self.features[filename][sentId]['history'])
                self.transformed_features[filename][sentId]['tense'] = self.Tenses.index(self.features[filename][sentId]['tense'])
                self.transformed_features[filename][sentId]['voice'] = self.Voices.index(self.features[filename][sentId]['voice'])
                self.transformed_features[filename][sentId]['modal'] = self.Modals.index(self.features[filename][sentId]['modal'])
        
    def getFeatures(self, filenames):
        X = []
        y = []
        for filename in filenames:
            for sentId in self.transformed_features[filename].keys():
                X.append(self.transformed_features[filename][sentId].values())
                y.append(self.transformed_features[filename][sentId]['val'])
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y

    def getSummary(self, filename):
        summary = []        
        for sentId in self.transformed_features[filename].keys():
            feature = self.transformed_features[filename][sentId].values()
            y = self.nb.predict([feature])
            if y in [1, 2, 4, 6]:
                summary.append(self.features[filename][sentId]['data'])
            if y in [0, 1, 5]:
                if random.uniform(0, 1) > 0.96:
                    summary.append(self.features[filename][sentId]['data'])        
        return "\n".join(summary)
                
    def train(self):
        if not self.isSummary:
            print "Train dataset: ", len(self.train_papers)
        self.reloadDis()        
        y_pred = self.nb.fit(self.train_X, self.train_y).predict(self.train_X)
        if not self.isSummary:
            print "Mislabelled sentences: " + str((self.train_y != y_pred).sum()) + " out of " + str(self.train_X.shape[0])
            print "Train Accuracy: " + str(self.accuracy((self.train_y != y_pred).sum(), self.train_X.shape[0]))

    def test(self, generate_histogram=False):        
        print "Test dataset length: ", len(self.test_papers)
        y_pred = self.nb.predict(self.test_X)
        if generate_histogram:
            plt.hist(y_pred, density=True)
            plt.savefig('histogram.png')
        print "Mislabelled sentences: " + str((self.test_y != y_pred).sum()) + " out of " + str(self.test_X.shape[0])
        print "Test Accuracy: " + str(self.accuracy((self.test_y != y_pred).sum(), self.test_X.shape[0]))
        # return self.getConfusionMatrix(self.test_y, y_pred)

    def accuracy(self, misclassifications, samples):
        return (1-(misclassifications/(samples*1.0)))*100.0

    def plotConfusionMatrix(self, cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')

    def getConfusionMatrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

if __name__ == '__main__':
    path_to_app_dir = '/'.join(__file__.split("/")[:-1])
    if path_to_app_dir:
        os.chdir(path_to_app_dir)
    Feature_vector = Features()
    folder = "../data/annotated_output"
    xmlfolder = "../data/tagged"
    Feature_vector.run(folder, xmlfolder)    

    print "=======Bernouli Distribution======="
    bnb = NaiveBayes(Feature_vector.feature_values, 0.8, "Bernoulli")    
    bnb.train()
    print ""    
    bnb.test()
    # confusionMatrix = bnb.test()
    # bnb.plotConfusionMatrix(confusionMatrix, range(8))
    print ""

    print "=======Multinomial Distribution======="
    mnb = NaiveBayes(Feature_vector.feature_values, 0.8, "Multinomial")
    mnb.train()
    print ""    
    mnb.test()
    # confusionMatrix = mnb.test()
    # mnb.plotConfusionMatrix(confusionMatrix, range(8))
    print ""

    print "=======Complement Distribution======="
    cnb = NaiveBayes(Feature_vector.feature_values, 0.8, "Complement")
    cnb.train()
    print ""    
    cnb.test()
    # confusionMatrix = cnb.test()
    # cnb.plotConfusionMatrix(confusionMatrix, range(8))    
    print ""

    print "=======Gaussian Distribution======="
    gnb = NaiveBayes(Feature_vector.feature_values, 0.8, "Gaussian")
    gnb.train()
    print ""
    gnb.test()    
    # confusionMatrix = gnb.test()
    # gnb.plotConfusionMatrix(confusionMatrix, range(8))    
    print ""