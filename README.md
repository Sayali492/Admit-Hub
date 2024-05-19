# AdmitHub: A Unified Platform for Predictive Analysis and Personalized Guidance in Postgraduate Admissions


A university recommender system offering admission chances for postgrad studies along with information about universities, admission process, courses, and scholarships, addressing student queries. Our algorithm  considers various factors like GRE score,toefl score,GPA,budget and other many other factors and recommends top 20 universities which are divided into "Safe","Ambitious" and "Moderate" universities. University Recommendations are done by utilizing a hybrid CF model by combining Neural Collaborative filtering(Multilayer Perceptron) and KNN. Here MLP model is used as it is unexplored in the field of recommendations but Deep learning has shown promising accuracy in other tasks. The model acheived 92.02% training and 73.74% testing accuracy. To improve the accuracy of our predictions further, we have incorporated a feedback loop.
Further, for overall admit percent prediction, customized ANN model is used.
For chatbot, ChatGPT3.5 API is used.


## Features

- University Recommendation
- Course Admit Percent
- Overall Admit Percent
- Chatbot for generalized queries
- Feedback


## Techstack

- Frontend: Flask
- Backend: Python
- Database: MongoDB
- Machine Learning Libraries- Keras, Tensorflow, sklearn, numpy, pandas
- API- GPT3.5 

## Research Gaps
Following the literature review, these were the gaps found and this project aims to overcome them.
1. Less availability of data
2. NCF is an unexplored alogrithm
3. No Feedback System
4. No Course Admit Percent recommended

## Dataset
The data has been scraped from  www.thegradcafe.com.The dataset had 25000 entries.We have merged two existing datasets,https://github.com/tramatejaswini/University_Recommendation_System and https://github.com/aditya-sureshkumar/University-Recommendation-System. After merging above datasets and filtering out the records that had result as "Admitted",finally the dataset had 58049 entries.The dataset is available at https://www.kaggle.com/datasets/anvitamahajan/university-recommendation-for-masters-phd-in-usa. 

## Methodology
For balancing the dataset, the data is resampled to make it balanced. After resampling, 28 Lakh entries are available in the dataset.
The university recommendation model is a hybrid model of Memory based and Model Based Collaborative
Filtering.Intially the model based CF is used,in which the Multi Layer Perceptron(MLP) model has been built.Then the memory based CF is used which uses K-Nearest Neighbors(KNN) Algorithm to determine the list of probable universities for the given profile. The architecture of model used is shown in the figure below:

![hybrid_model_arch2 drawio](https://github.com/Sayali492/Admit-Hub/assets/78889572/b9d09596-23ae-4775-8253-b6d3321c16e6)

For overall admit percent prediction, custom ANN model is built. Architecture is shown in diagram below:
![overall_admit_model_arch](https://github.com/Sayali492/Admit-Hub/assets/78889572/683d0e84-4363-437b-8fed-4e714ee220d5)


## Web Application Demo
The Demo is available at link: https://drive.google.com/file/d/1OWZzhkIgyNaHeb4Pf5DLOI_pcwnlMbXw/view?usp=sharing 


## Run Locally- UI

Clone the project

```bash
  git clone https://github.com/Sayali492/Admit-Hub/
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  flask run
```


## Authors

- [Anvita Mahajan](https://www.github.com/Anvita0305)
- [Sayali Mate](https://www.github.com/Sayali492)
- Chinmayee Kulkarni

