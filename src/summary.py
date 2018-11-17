from naive_bayes import NaiveBayes
from generate_feature import Features
import sys
import os
import pickle

if __name__ == '__main__':
    path_to_app_dir = '/'.join(__file__.split("/")[:-1])
    if path_to_app_dir:
        os.chdir(path_to_app_dir)
    filename = sys.argv[1]    
    distribution = "Gaussian"
    if len(sys.argv) > 2:
        distribution = sys.argv[2]
    
    if distribution == "Bernoulli":
        with open('bnb.pkl', 'rb') as fd:
            nb = pickle.load(fd)    
    elif distribution == "Complement":
        with open('cnb.pkl', 'rb') as fd:
            nb = pickle.load(fd)
    elif distribution == "Multinomial":
        with open('mnb.pkl', 'rb') as fd:
            nb = pickle.load(fd)
    else:
        with open('gnb.pkl', 'rb') as fd:
            nb = pickle.load(fd)
    print nb.getSummary(filename)
    
    

    # Feature_vector = Features()
    # folder = "../data/annotated_output"
    # xmlfolder = "../data/tagged"    
    # Feature_vector.run(folder, xmlfolder)
    # NB = NaiveBayes(Feature_vector.feature_values, 0.8, distribution, True)
    # NB.train()
    # print NB.getSummary(filename)

