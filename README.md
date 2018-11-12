## Summarization of Scientific Texts: A Rhetorical Approach

This project is the implementation of this [paper](https://www.mitpressjournals.org/doi/pdf/10.1162/089120102762671936)

The main idea behind this paper is to use a rhetorical approach for classifying different statements present in a scientific paper on basis of argumentative zoning.

This project builds towards automatic summarisation of scientific papers. We aim to classify each sentence within the research paper as one of the rhetorical categories.

### Annotation Based on Argumentative Zoning

Each of the statement in the paper is divided into following different categories-  
**1. Aim** - Specific research goal of the current paper  
**2. Textual** - Makes reference to the structure of the current paper    
**3. Own** - (Neutral) description of own work presented in current paper: Methodology, results, discussion  
**4. Background** - Generally accepted scientific background  
**5. Contrast** - Statements of comparison with or contrast to other work; weaknesses of other work  
**6. Basis** - Statements of agreement with other work or continuation of other work  
**7. Other** - (Neutral) description of other researchers’ work  

On basis of the above rhetorical categories we do the argumentative zoning of the sentences present in the papers. 

### Features Used

**1. Location** - Where in the document the sentence occurs  
**2. Section Structure** - Where in the section does the sentence occurs, i.e. if a sentence is a first line of the section and so on.  
**3. Paragraph Structure** - Whether a sentence occurs in the start, middle or the end of a paragraph.  
**4. Headline**  
**5. Length** - Whether the given line is a long line or not.  
**6. Title** - If the words in the sentence occur in the title or not.  
**7. Tf Idf Score** - Whether the sentence consists of significant words or not.  
**8. Voice** - What is the voice of the main verb of the sentence.  
**9. Tense** - Tense of the main verb or aux verb of the sentence.  
**10. Modal** - Just using the above concept we find whether there is an auxiliary verb with the main verb. If yes we give the corresponding values.  


### Approaches and Tools Used
We used existing argumentative zoning dataset and on that we created different feature vectors corresponding to each sentence, and then we trained a naive bayes classifier on the dataset. We did a test train split of 0.8

We used NLTK and Scikit for writing the classifier. Since we used scikit learn we were able to test our model with multiple distribution. 

We have used naive bayes with -
1. Bernoulli Distribution  
2. Gaussian Distribution  
3. Multinomial Distribution  
4. Complement Distribution  

### Results- 

- Total number of papers in the dataset - **79**
- Number of papers used for training - **64 ( 80% )**
- Number of papers used for testing - **15 ( 20% )**


#### Accuracies for classifying the sentences in to respective category
| Distribution      | Accuracy (in %) |
| -------------     | -------------   |
| **Bernoulli**     |     **84.64**   |
| **Gaussian**      |      **100**    |
| **Multinomial**   |     **80.89**   |
| **Complement**    |     **81.28**   |

**Plots:**  
### Bernoulli Distribution Confusion Matrix
![Bernoulli Distribution Confusion Matrix](https://github.com/pbteja1998/ire_project_18/raw/master/plots/bernouli_cf_matrix.png)
### Bernoulli Distribution Histogram
![Bernoulli Distribution Histogram](https://github.com/pbteja1998/ire_project_18/raw/master/plots/bernouli_hist.png)
### Complement Distribution Confusion Matrix
![Complement Distribution Confusion Matrix](https://github.com/pbteja1998/ire_project_18/raw/master/plots/complement_cf_matrix.png)
### Complement Distribution Histogram
![Complement Distribution Histogram](https://github.com/pbteja1998/ire_project_18/raw/master/plots/complement_hist.png)
### Gaussian Distribution Confusion Matrix
![Gaussian Distribution Confusion Matrix](https://github.com/pbteja1998/ire_project_18/raw/master/plots/guassian_cf_matrix.png)
### Gaussian Distribution Histogram
![Gaussian Distribution Histogram](https://github.com/pbteja1998/ire_project_18/raw/master/plots/guassian_hist.png)
### Multinomial Distribution Confusion Matrix
![Multinomial Distribution Confusion Matrix](https://github.com/pbteja1998/ire_project_18/raw/master/plots/multinomial_cf_matrix.png)
### Multinomial Distribution Histogram
![Multinomial Distribution Histogram](https://github.com/pbteja1998/ire_project_18/raw/master/plots/multinomail_hist.png)





### References

1. Argumentative Zoning https://arxiv.org/abs/1703.10152
2. https://www.cl.cam.ac.uk/~sht25/az.html
3. http://antonetteshibani.com/tag/argumentative-zoning/
