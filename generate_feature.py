import os,sys,re
import numpy as np
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords 
import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = Stemmer.Stemmer('english')
headlines_val = ["Introduction", "Implementation", "Example", "Conclusion","Result",
            "Evaluation","Solution","Discussion","Further Work","Data","Related Work",
            "Experiment","Problems","Method","Problem Statement","Non-Prototypical"]

class Features:

    def __init__(self):
        self.files = []
        self.feature_values = dict()
        self.tags = []
        self.stopwords = set(stopwords.words('english'))
        self.text = list()

    def insert_file(self, folder, xmlfolder, file):
        file_len = 0
        filepath1 = os.path.join(xmlfolder, file)
        filepath = os.path.join(folder, file)
        feature = dict()
        self.files.append(filepath1)
        with open(filepath,'r') as f:
            data = f.readlines()
            prev_his = 'BEGIN'
            new_text = list()
            for line in data:
                sen_feature = dict()
                line = line.strip('\n')
                if line == '':
                    continue
                file_len += 1
                sen_feature['data'] = line.split('\t')[1]
                sen_feature['val'] = line.split('\t')[0]
                sen_feature['num'] = file_len
                sen_feature['sid'] = line.split('\t')[2]
                sen_feature['history'] = prev_his
                prev_his = sen_feature['val']
                len1 = len(re.sub(r'[^0-9A-Za-z\_\-]', ' ' , sen_feature['data']).split())
                if len1 > 19:
                    sen_feature['len'] = 'YES'
                else:
                    sen_feature['len'] = 'NO'
                new_text.append(sen_feature['data'])
                feature[line.split('\t')[2]] = sen_feature
            self.feature_values[filepath1] = feature
            self.text.append(' '.join(new_text))


    def location(self):
        for key in self.feature_values:
            slen = len(self.feature_values[key]) / 20
            cur = 1
            for x in self.feature_values[key]:
                self.feature_values[key][x]['loc'] = chr(ord('A')+cur)
                if self.feature_values[key][x]['num'] % slen == 0:
                    cur+=1 

    def sec_location(self):
        for file in self.files:
            # print file
            tree = ET.parse(file)
            root = tree.getroot()
            for abstract in root.iter('ABSTRACT'):
                tot_lines = 1
                for i in abstract.iter('A-S'):
                    tot_lines += 1
                num = 1
                for sen in abstract.iter('A-S'):
                    sid = sen.attrib['ID']
                    try:
                        if num == 1:
                            self.feature_values[file][sid]['secloc'] = 'FIRST'
                        elif num == 2:
                            self.feature_values[file][sid]['secloc'] = 'SECOND'
                        elif num == 3:
                            self.feature_values[file][sid]['secloc'] = 'THIRD'
                        elif num == tot_lines-1:
                            self.feature_values[file][sid]['secloc'] = 'LAST'
                        elif num == tot_lines-2:
                            self.feature_values[file][sid]['secloc'] = 'SECOND-LAST'
                        elif num == tot_lines-3: 
                            self.feature_values[file][sid]['secloc'] = 'THIRD-LAST'
                        else:
                            self.feature_values[file][sid]['secloc'] = 'SOMEWHERE'
                    except KeyError as e:
                        continue
                    num += 1  

                tot_lines = 1
                for i in abstract.iter('A-S'):
                    tot_lines += 1
                num = 1
                for sen in abstract.iter('A-S'):
                    sid = sen.attrib['ID']
                    try:
                        if num == 1:
                            self.feature_values[file][sid]['parloc'] = 'INITIAL'
                        elif num == tot_lines-1:
                            self.feature_values[file][sid]['parloc'] = 'FINAL'
                        else:
                            self.feature_values[file][sid]['parloc'] = 'MEDIAL'
                    except KeyError as e:
                        continue
                    num += 1  

            for section in root.iter('DIV'):
                tot_lines = 1
                for i in section.iter('S'):
                    tot_lines += 1
                num = 1
                for sen in section.iter('S'):
                    sid = sen.attrib['ID']
                    try:
                        if num == 1:
                            self.feature_values[file][sid]['secloc'] = 'FIRST'
                        elif num == 2:
                            self.feature_values[file][sid]['secloc'] = 'SECOND'
                        elif num == 3:
                            self.feature_values[file][sid]['secloc'] = 'THIRD'
                        elif num == tot_lines-1:
                            self.feature_values[file][sid]['secloc'] = 'LAST'
                        elif num == tot_lines-2:
                            self.feature_values[file][sid]['secloc'] = 'SECOND-LAST'
                        elif num == tot_lines-3: 
                            self.feature_values[file][sid]['secloc'] = 'THIRD-LAST'
                        else:
                            self.feature_values[file][sid]['secloc'] = 'SOMEWHERE'
                    except KeyError as e:
                        continue
                    num += 1

            for para in root.iter('P'):
                tot_lines = 1
                for i in para.iter('S'):
                    tot_lines += 1
                num = 1
                for sen in para.iter('S'):
                    try:
                        sid = sen.attrib['ID']
                        if num == 1:
                            self.feature_values[file][sid]['parloc'] = 'INITIAL'
                        elif num == tot_lines-1:
                            self.feature_values[file][sid]['parloc'] = 'FINAL'
                        else:
                            self.feature_values[file][sid]['parloc'] = 'MEDIAL'
                    except KeyError as e:
                        continue
                    num += 1

            title = list()
            for tag in root.iter('TITLE'):
                text  = tag.text.split()
                for w in text:
                    w = stemmer.stemWord(w)
                    if w not in self.stopwords and  w not in title:
                        title.append(w)

            for sen in self.feature_values[file]:
                self.feature_values[file][sen]['Title'] = 'NO'
                text = self.feature_values[file][sen]['data'].split()
                for w in text:
                    w = stemmer.stemWord(w)
                    if w not in self.stopwords and w in title:
                        self.feature_values[file][sen]['Title'] = 'YES'
                        break

    def headlines(self):
        for file in self.files:
            tree = ET.parse(file)
            root = tree.getroot()
            for section in root.iter('DIV'):
                head_val = headlines_val[-1]
                for header in section.iter('HEADER'):
                    if header.text.strip() in headlines_val:
                        head_val = header.text.strip()
                    break

                for sen in section.iter('S'):
                    sid = sen.attrib['ID']
                    try:
                        self.feature_values[file][sid]['Headlines'] = head_val
                    except KeyError as e:
                        continue

            for sen in root.iter('A-S'):
                sid =  sen.attrib['ID']
                try:
                    self.feature_values[file][sid]['Headlines'] = 'Introduction'
                except KeyError as e:
                        continue

    def tfIdf(self):
        ''' '''
        vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english', ngram_range=(1, 1), norm='l2')
        vectorizer.fit_transform(self.text)
        indices = np.argsort(vectorizer.idf_)[::-1]
        feature_values = vectorizer.get_feature_names()
        tfidf = [feature_values[i] for i in indices[:18]]
        for file in self.feature_values:
            for sen in self.feature_values[file]:
                text = self.feature_values[file][sen]['data'].split()
                self.feature_values[file][sen]['tfidf'] = 'NO'
                for w in text:
                    if w in tfidf:
                        self.feature_values[file][sen]['tfidf'] = 'YES'
                        break

    def run(self, folder, xmlfolder):
        for file in os.listdir(folder):
            self.insert_file(folder, xmlfolder, file)
        self.location()
        self.sec_location()
        self.headlines()
        self.tfIdf()

if __name__ == '__main__':
    Feature_vector = Features()
    folder = sys.argv[1]
    xmlfolder = sys.argv[2]
    Feature_vector.run(folder, xmlfolder)