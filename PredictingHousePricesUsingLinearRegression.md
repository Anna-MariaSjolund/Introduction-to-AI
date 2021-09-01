# Predicting House Prices Using Linear Regression

Regardless of if you are contemplating selling your house, or if you are saving up to buy your dream home, you are probably interested in how much a similar house in a similar area is worth. In this paper I will describe the process of predicting future house prices, using linear regression. It will be loosely based on the checklist for machine learning projects, described by Aurélien Géron.<a href="#Geron(2017)" id="note1ref"><sup>1</sup></a>

## Frame the problem

We will use multiple linear regression to predict house prices, which falls into the category supervised machine learning.<a href="#Pant(2019)" id="note2ref"><sup>2</sup></a> We will use Python to create the models.

## Gathering the Data

There are already several available datasets for predicting house prices, such as the Boston Housing Dataset<a href="#Harrison&Rubinfeld(1978)" id="note3ref"><sup>3</sup></a> and the Ames Iowa Housing Data<a href="#deCock(2011)" id="note4ref"><sup>4</sup></a>, which includes variables such as: number of rooms, age of the house, proximity to larger roads,<a href="#Harrison&Rubinfeld(1978)" id="note3ref"><sup>3, </sup></a><a href="#deCock(2011)" id="note4ref"><sup>4</sup></a> and per capita crime rate<a href="#Harrison&Rubinfeld(1978)" id="note3ref"><sup>3</sup></a>. In the same manner as de Cock,<a href="#deCock(2011)" id="note4ref"><sup>4</sup></a> when he got access to the data by contacting the Ames City Assessor’s Office, I would suggest collecting the data from an authority, since it will be less biased, compared to for example data collected from a broker. In a Swedish context, Statistiska Centralbyrån holds classified microdata about prices and characteristics of houses in Sweden (Statistiska Centralbyrån, …) , that one can request to get access to in anonymised form (Statistiska Centralbyrån, …).  We would ideally have a sample size of … cases per independent variable (Tabachnick). 
I would store the data in an anonymised form in a csv(comma separated values)-file, which is commonly used for machine learning projects with tabular data (Jim Dowling). It is easy to store the data with a cloud service, for example Amazon SageMaker (reference). It is important to think about in which country the data is stored, if the dataset contains sensitive data, for example following GDPR for European countries.


## References
<a id="Geron(2017)" href="#note1ref">1</sup></a>. Géron, A. (2017)
*Hands-on machine learning with scikit-learn, Keras, and TensorFlow*. doi:10.5555/3153997 \
<a id="Pant(2019)" href="#note2ref">2</sup></a>. Pant, A. (2019, January 11). Workflow of a machine learning project. *Towards Data Science.* https://towardsdatascience.com/workflow-of-a-machine-learning-project-ec1dba419b94 \
<a id="Harrison&Rubinfeld(1978)" href="#note3ref">3</sup></a>. Harrison, D. & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. *Journal of Environmental Economics and Management, 5*(1), 81-102. https://doi.org/10.1016/0095-0696(78)90006-2 \
<a id="deCock(2011)" href="#note4ref">4</sup></a>. de Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data
as an end of semester regression project. *Journal of Statistics Education, 19*(3), 1-15. https://doi.org/10.1080/10691898.2011.11889627




