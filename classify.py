import spacy

# globals
nlp_spacy = spacy.load('en_core_web_sm')

data_file_path = "data/papers/"
annotated_index = "annotated_data"

def get_lines(filename, flines):
    lines = {}
    with open(data_file_path + filename) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

        for l in flines:
            lines[l] = content[l+1]

    return lines


def line_zone(ln):
    if ln >= 1 and ln <= 20:
        return 'A'
    elif ln >= 21 and ln <= 40:
        return 'B'
    elif ln >= 41 and ln <= 60:
        return 'C'
    elif ln >= 61 and ln <= 80:
        return 'D'
    elif ln >= 81 and ln <= 100:
        return 'E'
    elif ln >= 101 and ln <= 120:
        return 'F'
    elif ln >= 121 and ln <= 140:
        return 'G'
    elif ln >= 141 and ln <= 160:
        return 'H'
    else:
        return 'I'


def title_word(line, title):
    for w in line:
        if w in title:
            return True
    return False

def get_main_verb(line):
    pass


def citation(line):
    line = line.lower()
    if '[' or ']' or 'cit' or 'cite' or 'extend' or 'citation':
        return True
    return False


def long_sent(line):
    words = line.split(' ')
    if len(words) > 20:
        return True
    return False


def get_feature_vector(line, l_num, title):
    return [line_zone(l_num), title_word(line, title), citation(line), long_sent(line)]


content = []
with open(annotated_index) as af:
    content = af.readlines()

content = [x.strip() for x in content]

dataset = {}

filename = ""
field_name = ""
field_dict={}

for line in content:
    if line.startswith("`"):
        if filename != "":
            dataset[filename] = field_dict

        # reinitialize
        filename = line[1:]
        field_dict = {}
    elif line.startswith("#"):
        fields = line.split()
        field_name = fields[0][1:]

        if len(fields) > 1:
            field_data = fields[1].split(",")
            field_data = [int(i) for i in field_data]
            field_dict[field_name] = field_data
        else:
            field_dict[field_name] = []

# lets load up the lines now
datalines = {}
for f in dataset.keys():
    all_lines = []
    for k in dataset[f].keys():
        all_lines += dataset[f][k]
    datalines[f] = get_lines(f, all_lines)

# calculate counts of all the classes & total sentences
total_sentences = 0
categ_sents = {
        "AIM": 0,
        "TEXTUAL": 0,
        "OWN": 0,
        "BACKGROUND": 0,
        "CONTRAST": 0,
        "BASIS": 0,
        "OTHER": 0
}

for f in dataset.keys():
    rhet_categs = dataset[f].keys()

    for rc in rhet_categs:
        total_sentences += len(dataset[f][rc])
        categ_sents[rc] += len(dataset[f][rc])

# FEATURE KEY
#
# LOC: Location in the document
# LEN: Long/Short > 12 words, LONG
# TITLE: Does the sentence contain the words appearing in title
# CIT: Does the sentence contain any citation references
#

feature_matrix = {
        "LOC": [],
        "LEN": [],
        "TITLE": [],
        "CIT": [],
}
