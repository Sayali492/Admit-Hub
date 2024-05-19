# AdmitHub: A Unified Platform for Predictive Analysis and Personalized Guidance in Postgraduate Admissions


A university recommender portal offering admission chances for postgrad studies along with information about universities, admission process, courses, and scholarships, addressing student queries.Our portal considers various factors like GRE score,toefl score,GPA,budget and other many other factors and recommends top 20 universities which are divided into "Safe","Ambitious" and "Moderate" universities.

## Dataset
The data has been scraped from  www.thegradcafe.com.The dataset had 25000 entries.We have merged two existing datasets,https://github.com/tramatejaswini/University_Recommendation_System and https://github.com/aditya-sureshkumar/University-Recommendation-System. After merging above datasets and filtering out the records that had result as "Admitted",finally the dataset had 58049 entries.The dataset is available at https://www.kaggle.com/datasets/anvitamahajan/university-recommendation-for-masters-phd-in-usa. 
## Methodology
The university recommendation model is a hybrid model of Memory based and Model Based Collaborative
Filtering.Intially the model based CF is used,in which the Multi Layer Perceptron(MLP) model has been built.Then the memory based CF is used which uses K-Nearest Neighbors(KNN) Algorithm to determine the list of probable universities for the required profile.
## Web Application
The Web aplication is created using python Flask.The demo of the UI is available in "UI Demo.mp4".
