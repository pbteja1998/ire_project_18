from naive_bayes import NaiveBayes
from generate_feature import Features
import pickle
import os

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
with open('bnb.pkl', 'wb') as fid:
    pickle.dump(bnb, fid, -1)
bnb.test()
# confusionMatrix = bnb.test()
# bnb.plotConfusionMatrix(confusionMatrix, range(8))
print ""

print "=======Multinomial Distribution======="
mnb = NaiveBayes(Feature_vector.feature_values, 0.8, "Multinomial")
mnb.train()
print ""
with open('mnb.pkl', 'wb') as fid:
    pickle.dump(mnb, fid, -1)
mnb.test()
# confusionMatrix = mnb.test()
# mnb.plotConfusionMatrix(confusionMatrix, range(8))
print ""

print "=======Complement Distribution======="
cnb = NaiveBayes(Feature_vector.feature_values, 0.8, "Complement")
cnb.train()
print ""
with open('cnb.pkl', 'wb') as fid:
    pickle.dump(cnb, fid, -1)
cnb.test()
# confusionMatrix = cnb.test()
# cnb.plotConfusionMatrix(confusionMatrix, range(8))    
print ""

print "=======Gaussian Distribution======="
gnb = NaiveBayes(Feature_vector.feature_values, 0.8, "Gaussian")
gnb.train()
print ""
gnb.test()
with open('gnb.pkl', 'wb') as fid:
    pickle.dump(gnb, fid, -1)
# confusionMatrix = gnb.test()
# gnb.plotConfusionMatrix(confusionMatrix, range(8))    
print ""