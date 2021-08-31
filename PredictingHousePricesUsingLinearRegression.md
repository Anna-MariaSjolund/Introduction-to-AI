# Predicting House Prices Using Linear Regression

Regardless of if you are contemplating selling your house, or if you are saving up to buy your dream home, you will be interested in how much a similar house in a similar area is worth. In this paper I will describe the process of predicting future house prices, using linear regression. It will be loosely based on the steps for machine learning projects, described by Aurélien Géron<a href="#note1" id="note1ref"><sup>1</sup></a>.

## Frame the problem

We will use multiple linear regression to predict house prices, which falls into the category supervised machine learning. We will use Python to create the models.

## Gathering the Data

There are already several available datasets for predicting house prices, such as Boston Housing Dataset (Harrison & Rubinfeld, 1978) and the Ames Iowa Housing Data (De Cock, 2011), which includes variables such as: number of rooms, age of the house, proximity to larger roads, (Harrison & Rubinfeld, 1978; De Cock, 2011) and per capita crime rate (Harrison & Rubinfield, 1978). In the same manner as de Cock, when he (2011) got access to the data by contacting the Ames City Assessor’s Office, I would suggest collecting the data from an authority, since it will be less biased, compared to for example data collected from a broker. In a Swedish context, Statistiska Centralbyrån holds classified microdata about prices and characteristics of houses in Sweden (Statistiska Centralbyrån, …) , that one can request to get access to in anonymised form (Statistiska Centralbyrån, …).  We would ideally have a sample size of … cases per independent variable (Tabachnick). 
I would store the data in an anonymised form in a csv(comma separated values)-file, which is commonly used for machine learning projects with tabular data (Jim Dowling). It is easy to store the data with a cloud service, for example Amazon SageMaker (reference). It is important to think about in which country the data is stored, if the dataset contains sensitive data, for example following GDPR for European countries.


## References
<a id="note1" href="#note1ref">1</sup></a>. Gero 2019
