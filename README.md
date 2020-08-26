# ds_audiobooks_predictor
Case example for predicting future behaviour of customers with a dataset of audiobook-sales. Goal is to be able to predict wether a customer is likely to buy again in the future, based on various factors.

Dataset
---------
The dataset is taken from this udemy course:  
https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/  
`Audiobooks_new_data.csv` has descriptive labels outlining all relevant datapoints that were used  
TLDR: the dataset lists users and their past bahaviour with the audiobook service and indicates wether said users did buy there again later

Preprocessing steps
---------
* balancing
* shuffling
* standardization
* splitting into train, validate and test sets (80/10/10)
`Audiobooks_preprocessing.ipynb`

Model
---------
a pretty basic, 3-layer tensorflow fnn model was trained on the data and later used to predict new, similar one  
loss-function: categorical crossentropy, optimizer: adam (with adjusted learning rate)
`Audiobooks_model.ipynb`  
new data is preprocessed and used for predictions in the same way the original data was by using the functions of the audiobooks-module `audiobooks_module.py` which uses all previously serialized support files necessary to make predictions in a consistent way and returns a dataframe that only contains relevant datapoints and additionally also the predictions of the deployed model, hence in a ready-to-analyze and visualize-form (e.g. with tableau)  
The interaction with said module can be seen here: `Audiobooks_predict_new_data.ipynb`

Visualization
---------
![Tableau_Example](images/Dashboard_audiobooks.png?raw=true "example of tableau visualizations")
Those visualizations were created by 100 new customer-rows which were predicted by the model and show the likelyhood of customers buying again in the future. They were created with tableau after the creation of an analyzable table (`Audiobook_predictions.csv`)  
They allow us to make a few assumptions based on the graphs we see:

Probability vs Completion:  
* it is more likely for customers with low completion rates of their audiobooks to buy again 
* it seems to be more likely for customers who spend less money on the service so far to buy again

Probability vs Review:   
* customers who gave better reviews are slightly more likely to buy again
* customers who give better reviews generally have higher completion rates and spend more on the service, those datapoints seem to be correlated

Probability vs Avg. Price:  
* it is more likely for a customer to buy again the lower the average price of his purchased books is
* we can see that it is more likely for users who have low completion rates to buy again here too

Probability vs Time since last purchase:
* it gets more likely for a customer to buy again the longer it has been since his last purchase
* we see that low completion rates and low overall spendings are a driving factor once again