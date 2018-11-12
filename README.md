## Summarization of Scientific Texts: A Rhetorical Approach

This project is the implementation of this [paper](https://www.mitpressjournals.org/doi/pdf/10.1162/089120102762671936)

The main idea behind this paper is to use a rhetorical approach for classifying different statements present in a scientific paper on basis of argumentative zoning.

This project builds towards automatic summarisation of scientific papers. We aim to classify each sentence within the research paper as one of the rhetorical categories.

### Annotation Based on Argumentative Zoning

Each of the statement in the paper is divided into following different categories-
**1. Aim** - Specific research goal of the current paper
**2. Own** - (Neutral) description of own work presented in current paper: Methodology, results, discussion
**3. Background** - Generally accepted scientific background
**4. Contrast** - Statements of comparison with or contrast to other work; weaknesses of other work
**5. Basis** - Statements of agreement with other work or continuation of other work
**6. Other** - (Neutral) description of other researchers’ work

On basis of the above rhetorical categories we do the argumentative zoning of the sentences present in the papers. 

### Features Used

**1. Location** - Where in the document the sentence occurs
**2. Section Structure** - Where in the section does the sentence occurs, i.e. if a sentence is a first line of the section and so on.
**3. Paragraph Structure** - Whether a sentence occurs in the start, middle or the end of a paragraph.
**4. Headline**
**5. Length** - Whether the given line is a long line or not.
**6. Title** - If the words in the sentence occur in the title or not.
**7. Tf Idf Score** - Whether the sentence consists of significant words or not.
**8. Voice** - What is the voice of the main verb of the sentence
**9. Tense** - Tense of the main verb or aux verb of the sentence.
**10. Modal** - Just using the above concept we find whether there is an auxiliary verb with the main verb. If yes we give the corresponding values.


### Approaches and Tools Used
We used existing argumentative zoning dataset and on that we created different feature vectors corresponding to each sentence, and then we trained a naive bayes classifier on the dataset. We did a test train split of 0.8

We used NLTK and Scikit for writing the classifier. Since we used scikit learn we were able to test our model with multiple distribution. 

We have used naive bayes with -
1. Bernoulli Distribution
2. Gaussian Distribution
3. Multinomial Distribution
4. Compliment Distribution

Results- 

Our observations are [here](https://scontent-bom1-2.xx.fbcdn.net/v/t1.15752-9/46007523_347829612450454_2516439562272636928_n.png?_nc_cat=104&_nc_ht=scontent-bom1-2.xx&oh=f7ba6a93ca984bf683b7b2bf6e364b95&oe=5C3E02E5)


### References

1. Argumentative Zoning https://arxiv.org/abs/1703.10152
2. https://www.cl.cam.ac.uk/~sht25/az.html
3. http://antonetteshibani.com/tag/argumentative-zoning/
