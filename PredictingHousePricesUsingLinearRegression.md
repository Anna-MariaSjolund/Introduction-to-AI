# Predicting House Prices Using Linear Regression

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Regardless of if you are contemplating selling your house, or if you are saving up to buy your dream home, you are probably interested in how much a similar house in a similar area is worth. In this paper I will describe the process of predicting house prices, using linear regression. It will be loosely based on the checklist for machine learning projects, described by Aurélien Géron.<a href="#Geron(2017)" id="note1ref"><sup>1</sup></a>

## Frame the Problem

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Multiple linear regression, which falls into the category supervised machine learning,<a href="#Pant(2019)" id="note2ref"><sup>2</sup></a> will be used to predict the house prices. Using this method, de Cock and colleagues<a href="#deCock(2011)" id="note3ref"><sup>3</sup></a> was able to explain between 80% (using two independent variables (IV)) and 92% (using 36 IV:s) of the variation in price. It would therefore be reasonable to aim to explain somewhere between 85-90% of the variance. In this paper, I will focus on how to create the model in Python, although it would also be possible to use other programming languages, such as R.

## Gathering the Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;There are already several available datasets for predicting house prices, e.g. the Boston Housing Dataset<a href="#Harrison&Rubinfeld(1978)" id="note4ref"><sup>4</sup></a> and the Ames Iowa Housing Data<a href="#deCock(2011)" id="note3ref"><sup>3</sup></a>, which include variables such as: number of rooms, age of the house, proximity to larger roads,<a href="#deCock(2011)" id="note3ref"><sup>3,</sup></a> <a href="#Harrison&Rubinfeld(1978)" id="note4ref"><sup>4 </sup></a>and per capita crime rate<a href="#Harrison&Rubinfeld(1978)" id="note4ref"><sup>4</sup></a>. I would suggest collecting the data from an authority, since it will probably be less biased, compared to for example data collected from a broker. In a Swedish context, Statistiska Centralbyrån holds classified microdata about prices and characteristics of houses in Sweden,<a href="#StatistiskaCentralbyran(2021)a" id="note5ref"><sup>5</sup></a> which one can request to get access to in anonymised form.<a href="#StatistiskaCentralbyran(2021)b" id="note6ref"><sup>6</sup></a> We would ideally have a sample size of *N* ≥ 50 + 8*m* (*m* is the number of IV:s) for testing the full model and *N* ≥ 104 + *m* for testing the individual predictors.<a href="#Tabachnick&Fidell(2014)" id="note7ref"><sup>7</sup></a> \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Once the data is collected I would recommend to initially save the data in an anonymised form in a csv-file, which is commonly used for machine learning projects with tabular data.<a href="#Dowling(2019)" id="note8ref"><sup>8</sup></a> The data can then be stored in the cloud, using for example Amazon SageMaker.<a href="#AmazonWebServices,Inc(2019)" id="note9ref"><sup>9</sup></a> It is also important to follow local regulations for data storage, for example GDPR.<a href="#EuropeanCommision(n.d.)" id="note10ref"><sup>10</sup></a>

## Exploring and Preparing the Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Once the data is collected, it is time to do some initial exploration. For example, Matplotlib<a href="#Hunter(2007)" id="note11ref"><sup>11</sup></a> can be used to create simple boxplots for visually identifying outliers and to make scatterplots for getting an overview of the linear relationship between the variables. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the next step we have to prepare the raw data for analysis by cleaning it, using for example Pandas<a href="#McKinney(2010)" id="note12ref"><sup>12</sup></a> or NumPy.<a href="#HarrisEtAl(2020)" id="note13ref"><sup>13</sup></a> If the dataset contains missing values we can either remove the whole row, or we can replace the missing value, with for example the mean.<a href="#Pant(2019)" id="note2ref"><sup>2</sup></a> We should also remove previously identified outliers and check for collinearity/multicollinearity, as well as transform the data if it is not normally distributed.  Furthermore, if the dataset contains discrete values, they can be converted to a set of dichotomous values, using dummy-variable coding.<a href="#Tabachnick&Fidell(2014)" id="note7ref"><sup>7</sup></a> \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The dataset may have to be converted to a NumPy array, if the analysis will be conducted in for example scikit-learn<a href="#PedregosaEtAl(2011)" id="note14ref"><sup>14</sup></a>. The dataset should be randomly split into *training* for training the model, *validation* for tuning the parameters and *testing* for testing the model on unseen data.<a href="#Pant(2019)" id="note2ref"><sup>2</sup></a> This can be done with train_test_split from scikit-learn. However, if we use an analytical solution to create the model, the validation dataset will not be used.<a href="#Geron(2017)" id="note1ref"><sup>1</sup></a> A common way to split the data is to use 25-30% for testing and 70-75% for training.<a href="#Mueller&Massaron(n.d.)" id="note15ref"><sup>15</sup></a> If the dataset is small, we can use cross-validation, which runs the model numerous times on different subsets of the data.<a href="#Cook(n.d.)" id="note16ref"><sup>16</sup></a>

## Creating the Model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Since the dataset probably have several IV:s, multiple linear regression can be used. The formula for this is: Y’= A + B1X1 + B2X2 + … + BkXk. Y is the dependent variable (DV), A is the intercept, the X:s represent the IV:s and the B:s are the regression coefficients. The aim of this statistical method is to determine the B-values for the IV:s, which summarize to a Y-value (i.e. price), which should be as close as possible to the actual price. The final model should therefore predict the price of the house, based on the X-values that we input.<a href="#Tabachnick&Fidell(2014)" id="note7ref"><sup>7</sup></a>  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Depending on the complexity of the dataset, we can choose to either create the model using an analytical solution (e.g. linear_model from scikit-learn) or we can solve it with gradient descent (e.g. SGDRegressor class from scikit-learn), that iterates over the training set, updates the model accordingly and uses the validation set for an unbiased evaluation of the model’s fit. The latter is significantly faster for larger datasets.<a href="#Geron(2017)" id="note1ref"><sup>1</sup></a>

## Fine-Tune the Model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Once we have generated the first model, we should fine-tune the parameters, changing for example the learning rate if using gradient descent or adding or removing variables if taking an analytical approach. We should also make sure that we did not overfit the model, i.e. that the model is too closely aligned to the training set and cannot generalise to the validation data. After this step we can compare our model to the test data and get a sense of how it will work in real life.<a href="#Geron(2017)" id="note1ref"><sup>1</sup></a>

## Present the Model 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We can use seaborn<a href="#Waskom(2021)" id="note17ref"><sup>17</sup></a> to create scatterplots showing how the IV:s are related to the DV. Depending on the number of IV:s, we can either show the regression lines for each IV on one plot, or we can create separate plots. For showing how neighbourhood and price are related we can create a heatmap showing price in different colours. In Jupyter notebooks we can use gmaps, which is a plugin that allows us to embed Google Maps directly into the notebook.<a href="#Bugnion(2016)" id="note18ref"><sup>18</sup></a>

## Deploy the Model 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;One way to deploy the model is to do as Dabreo and colleagues, who created a user interface with Flask technology, where the user can input features of the house and in return receive an estimated price.<a href="#DabreoEtAl(2021)" id="note19ref"><sup>19</sup></a> For a more advanced configuration we can use AWS SageMaker. Finally, we have to continuously update our model with new data (using for example A/B testing, where we evaluate if the new model is better and in that case gradually shift traffic to the new model<a href="#Kavanagh(2020)" id="note20ref"><sup>20</sup></a>), since predictors such as the attractiveness of the area and the general house market will change. 

## References
<a id="Geron(2017)" href="#note1ref">1</sup></a>. Géron, A. (2017). *Hands-on machine learning with scikit-learn, Keras, and TensorFlow*. O'Reilly Media, Inc. 

<a id="Pant(2019)" href="#note2ref">2</sup></a>. Pant, A. (2019, January 11). Workflow of a machine learning project. *Towards Data Science.* https://towardsdatascience.com/workflow-of-a-machine-learning-project-ec1dba419b94 

<a id="deCock(2011)" href="#note3ref">3</sup></a>. de Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an end of semester regression project. *Journal of Statistics Education, 19*(3), 1-15. https://doi.org/10.1080/10691898.2011.11889627 

<a id="Harrison&Rubinfeld(1978)" href="#note4ref">4</sup></a>. Harrison, D. & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. *Journal of Environmental Economics and Management, 5*(1), 81-102. https://doi.org/10.1016/0095-0696(78)90006-2 

<a id="StatistiskaCentralbyran(2021)a" href="#note5ref">5</sup></a>. Statistiska Centralbyrån. (n.d.). *SCB:s mikrodataregister.* Retrieved September 01, 2021, from https://www.h6.scb.se/metadata/mikrodataregister.aspx 

<a id="StatistiskaCentralbyran(2021)b" href="#note6ref">6</sup></a>. Statistiska Centralbyrån. (n.d.). *Beställa mikrodata.* Retrieved September 01, 2021, from https://scb.se/vara-tjanster/bestall-data-och-statistik/bestalla-mikrodata/

<a id="Tabachnick&Fidell(2014)" href="#note7ref">7</sup></a>. Tabachnick, B.G. & Fidell, L.S. (2014). *Using multivariate statistics*. Pearson Education Limited.

<a id="Dowling(2019)" href="#note8ref">8</sup></a>. Dowling, J. (2019, October 25). Guide to file formats for machine learning: columnar, training, inferencing, and the feature store. *Towards Data Science.* https://towardsdatascience.com/guide-to-file-formats-for-machine-learning-columnar-training-inferencing-and-the-feature-store-2e0c3d18d4f9 

<a id="AmazonWebServices,Inc(2019)" href="#note9ref">9</sup></a>. Amazon Web Services, Inc. (2019). *Amazon SageMaker, Machine learning for every data scientist and developer.* https://aws.amazon.com/sagemaker/ 

<a id="EuropeanCommision(n.d.)" href="#note10ref">10</sup></a>. European Commision. (n.d.). Data protection in the EU, The General Data Protection Regulation (GDPR), the Data Protection Law Enforcement Directive and other rules concerning the protection of personal data. Retrieved September 01, 2021, from https://ec.europa.eu/info/law/law-topic/data-protection/data-protection-eu_en 

<a id="Hunter(2007)" href="#note11ref">11</sup></a>. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

<a id="McKinney(2010)" href="#note12ref">12</sup></a>. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference, 445*, 56-61. https://doi.org/10.25080/Majora-92bf1922-00a

<a id="HarrisEtAl(2020)" href="#note13ref">13</sup></a>. Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D.,...Oliphant, T.E. (2020). Array programming with NumPy. *Nature, 585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2

<a id="PedregosaEtAl(2011)" href="#note14ref">14</sup></a>. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,...Duchesnay, E. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830. https://scikit-learn.org/stable/faq.html

<a id="Mueller&Massaron(n.d.)" href="#note15ref">15</sup></a>. Mueller, J.P. & Massaron, L. (n.d.). Training, validating, and testing in machine learning. *Dummies A Wiley Brand.* Retrieved September 01, 2021, from https://www.dummies.com/programming/big-data/data-science/training-validating-testing-machine-learning/ 

<a id="Cook(n.d.)" href="#note16ref">16</sup></a>. Cook, A. (n.d.) Cross-validation. *Kaggle.* Retrieved September 01, 2021, from https://www.kaggle.com/alexisbcook/cross-validation

<a id="Waskom(2021)" href="#note17ref">17</sup></a>. Waskom, M.L. (2021). Seaborn: statistical data visualization. *Journal of Open Source Software, 6*(60), 3021. https://doi.org/10.21105/joss.03021

<a id="Bugnion(2016)" href="#note18ref">18</sup></a>. Bugnion, P. (2016). Getting started. *jupyter-gmaps.* Retrieved September 02, 2021, from https://jupyter-gmaps.readthedocs.io/en/latest/tutorial.html

<a id="DabreoEtAl(2021)" href="#note19ref">19</sup></a>. Dabreo, S., Rodrigues, S., Rodrigues, V. & Shah, P. (2021). Real estate price prediction. *International Journal of Engineering Research & Technology, 10*(4), 644-649. Retrieved from https://www.ijert.org/research/real-estate-price-prediction-IJERTV10IS040322.pdf

<a id="Kavanagh(2020)" href="#note20ref">20</sup></a>.
Kavanagh, K., Nigenda, D. & Pydi, A. (2020, June 12). A/B Testing ML models in production using Amazon SageMaker. *Amazon Web Services, Inc.* https://aws.amazon.com/blogs/machine-learning/a-b-testing-ml-models-in-production-using-amazon-sagemaker/