import json
import os

from flask import Flask, render_template, request,session,redirect, url_for, flash, send_from_directory
from sklearn.discriminant_analysis import StandardScaler
from forms import InputForm
from feedback import FeedbackForm
from courseDetailsForm import InputCourseForm
from actualAdmits import ActualAdmitsInputForm
from tensorflow.keras.models import load_model
import secrets
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
from openai import OpenAI
from flask import jsonify
from pymongo import MongoClient
from flask import g
from g4f.client import Client
import tkinter
from tkinter import messagebox
from langdetect import detect
from translate import Translator
# from flask_httpauth import HTTPBasicAuth

# api_key = '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'
api_key='sk-proj-9S3vy7pLHOn5r4adqrUOT3BlbkFJGHKxsbBR3viAFSL4nRms'
# chatgpt_client = OpenAI(api_key=api_key)
chatgpt_client = Client()
model = load_model('Models/nn_for_uni_rec_wt_toefl.h5')
overallAdmitModel = load_model('Models/Overall_admit_percent_updated (1).h5')
courseModel = load_model('Models/nn_for_course_rec (1).h5')
client = MongoClient('mongodb+srv://sayalimate02:JELDh0YK2FZWfkye@admithub.zbzzdrj.mongodb.net/?retryWrites=true&w=majority&appName=AdmitHub')
db = client['AdmitHub']  # Replace with your MongoDB database name
users_collection = db['users']
predicted_admits=db['predicted_admits']
actual_admits=db['actual_admits']
translator= Translator(from_lang="chinese",to_lang="english")

# dataset paths
unresampled_dataset = "Data Files/final dataset using knn.csv"
resampled_dataset = "Data Files/final dataset after midterm.csv"
admit_pred_dataset="Data Files/Admission_Predict.csv"
app = Flask(__name__)
secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] = secret_key  # Set a secret key for CSRF protection

# auth = HTTPBasicAuth()

@app.route('/get_json')
def get_json():
    # Replace 'static' with the directory where your JSON file is located
    return send_from_directory('static', 'university_courses_latest.json')


@app.route("/home")
def view_home():
    return render_template("index.html", title="Home page")

def overall_admit_percent(gre, toefl, rating, GPA, Research):
    df = pd.read_csv(admit_pred_dataset)

    # Convert LOR and SOP to numerical ratings
    # lor_rating = generate_response(LOR + " Rate this LOR out of 10. Give only number")
    # sop_rating = generate_response(SOP + " Rate this SOP out of 10. Give only number")

    # Create DataFrame with the appropriate structure
    df=df[['GRE Score','TOEFL Score','University Rating','CGPA','Research']]
    # print(df)
    data = [gre, toefl, rating, GPA, Research]
    columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA', 'Research']
    new_df = pd.DataFrame([data], columns=columns)
    # print(new_df)
        # Concatenate the original DataFrame with the new row
    df = pd.concat([df, new_df], ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Select the last row after scaling
    new_row_scaled = X_scaled[-1].reshape(1, -1)  # Reshape for prediction

    # Predict using the model
    print(new_row_scaled)
    prediction = overallAdmitModel.predict(new_row_scaled)
    return prediction

# for recommending universities
@app.route("/rec_uni", methods=['GET', 'POST'])
def view_first_page():
    options = []
    username = session.get('username')
    print(username)
    with open('Data Files/course_rankings.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            options.append((parts[1], parts[0]))  # (value, label)

    form = InputForm()
    print(form.validate_on_submit())
    if form.validate_on_submit():
        print('in the form')
        gpa = form.gpa.data
        gre_awa = form.gre_awa.data
        gre_verbal = form.gre_verbal.data
        gre_quant = form.gre_quant.data
        toefl = form.toefl.data
        publication = form.publication.data
        re_exp = form.re_exp.data
        work_exp = form.work_exp.data
        budget = form.budget.data
        course = form.course.data
        scale = form.scale.data
        degree = form.degree.data
        lor = form.degree.data
        sop = form.degree.data
        rank= form.rank.data
        print(gre_awa, gre_quant, gre_verbal, gpa, course)

        # preprocessing the input
        df = pd.read_csv(unresampled_dataset)
        re_df = pd.read_csv(resampled_dataset)

        numerical_columns = ['toeflScore', 'GRE_AWA', 'GRE_QUANT', 'GRE_VERBAL', 'GPA', 'min_gpa', 'gre_total']

        for col in numerical_columns:
            if col == 'min_gpa' or col == 'GPA' or col == 'GRE_AWA':
                df[col] = df[col].astype(float).round(2)
            else:
                df[col] = df[col].astype(int)
        df = df.groupby('Course Name').filter(lambda x: len(x) >= 50)

        profile = {
            'toeflScore': toefl,
            'GRE_AWA': gre_awa,
            'GRE_QUANT': gre_quant,
            'GRE_VERBAL': gre_verbal,
            'GPA': gpa,
            'Publications': publication,
            'Work experience': work_exp,
            'Research Experience': re_exp,
            'Course Name Encoded': course,
            'scale': scale,
            'Degree': degree,
            # 'LOR': lor,
            # 'SOP': sop,
            'Budget': budget
        }

        # weights
        weights = {
            "toeflScore": 0.96609756,
            "GRE_AWA": 0.72707317,
            "GRE_QUANT": 0.84658537,
            "GRE_VERBAL": 0.99,
            "GPA": 0.70317073,
            "Course Name Encoded": 0.72707317,
        }
        column_types = re_df.dtypes
        print(column_types)

        re_df['Score'] = re_df['toeflScore'] * weights['toeflScore'] + re_df['GRE_AWA'] * weights['GRE_AWA'] + re_df[
            'GRE_QUANT'] * weights['GRE_QUANT'] + re_df['GRE_VERBAL'] * weights['GRE_VERBAL'] + re_df['GPA'] * weights[
                             'GPA']
        min_score = re_df['Score'].min()

        # Normalize the Score column
        re_df['Normalized Score'] = re_df['Score'] - min_score

        profile_df = pd.DataFrame([profile])
        # print(profile_df)

        # GRE scores processing
        profile_df['GRE_QUANT'] = profile_df['GRE_QUANT'] - 130
        profile_df['GRE_VERBAL'] = profile_df['GRE_VERBAL'] - 130

        # GPA conversion function
        def convert_gpa(gpa, scale):
            if scale == 10:
                return gpa / 2.5
            elif scale == 5:
                return (gpa / 5) * 4
            elif scale == 20:
                return gpa / 5.0
            elif scale == 100:
                return gpa / 25.0
            else:
                return gpa

                # Convert GPA column to a 4.0 scale

        admit_percent_overall=overall_admit_percent(gre_quant+gre_verbal,toefl,rank,gpa,re_exp)
        admit_percent_overall=admit_percent_overall[0][0].round(4)*100
        admit_percent_overall=admit_percent_overall.round(2)
        print("Overall Admit Percent:",admit_percent_overall)

        profile_df['GPA'] = profile_df.apply(lambda row: convert_gpa(row['GPA'], scale=scale), axis=1)

        # Concatenate
        df = pd.concat([df, profile_df], ignore_index=True)
        df.drop(['min_gpa', 'gre_total', 'min_gre'], axis=1, inplace=True)

        # Scaling the profile using RobertScalar
        scaler = RobustScaler()
        columns = ['GPA', 'GRE_AWA', 'GRE_QUANT', 'GRE_VERBAL', 'toeflScore', 'Publications', 'Research Experience',
                   'Work experience']
        df[columns] = scaler.fit_transform(df[columns])
        ip_df = df.iloc[[-1]]

        # final preprocessed df for input
        input = ip_df[['GRE_AWA', 'GRE_QUANT', 'GRE_VERBAL', 'GPA', 'Course Name Encoded']]
        logits = model.predict(input)  # Replace with actual prediction function

        # Extract probabilities and corresponding class labels
        probabilities = logits.ravel()  # Flatten the array
        class_labels = np.arange(len(probabilities))  # Assuming class indices start from 0

        # Get indices of top 10 predicted classes
        top_10_indices = np.argsort(probabilities)[::-1][:40]

        # Get corresponding probabilities and class labels for top 10
        top_10_probabilities = probabilities[top_10_indices]
        top_10_class_labels = class_labels[top_10_indices]

        # class to university
        def create_class_to_university_dict(file_path):
            class_to_university = {}
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        class_number = int(parts[1].strip())
                        university_name = parts[0].strip()
                        class_to_university[class_number] = university_name
            return class_to_university

        class_to_university = create_class_to_university_dict('Data Files/university_rankings.txt')

        # Display top 10 predicted classes along with their probabilities
        predictions = []
        for i in range(20):
            filtered_df = re_df[re_df['University Name Encoded'] == top_10_class_labels[i]]
            max_score = filtered_df['toeflScore'].max() * weights['toeflScore'] + \
                        filtered_df['GRE_AWA'].max() * weights['GRE_AWA'] + \
                        filtered_df['GRE_QUANT'].max() * weights['GRE_QUANT'] + \
                        filtered_df['GRE_VERBAL'].max() * weights['GRE_VERBAL'] + \
                        filtered_df['GPA'].max() * weights['GPA']
            score = ip_df['GRE_AWA'] * weights['GRE_AWA'] + \
                    ip_df['GRE_QUANT'] * weights['GRE_QUANT'] + \
                    ip_df['GRE_VERBAL'] * weights['GRE_VERBAL'] + \
                    ip_df['GPA'] * weights['GPA']
            admit_percent = ((score - min_score) / (max_score - min_score)) * 100
            admit_percent=admit_percent.round(2)
            predictions.append([top_10_class_labels[i], top_10_probabilities[i], admit_percent.tolist()[0]])
            print("Class:", top_10_class_labels[i], " Probability:", top_10_probabilities[i], " Admit Percent:",
                  admit_percent)

            # Map class number to university name
            university_name = class_to_university.get(top_10_class_labels[i])
            predictions[-1].append(university_name) 
             # Append university name to the last prediction
        predictions=sorted(predictions,key=lambda x:x[2])
        predictions=predictions[::-1]


        pred_unis=[]
        for prediction in predictions:
            pred_unis.append(prediction[3])
        #insert in database
        if users_collection.find_one({'username': username}):
            predicted_admits.insert_one({
                'username': username, 
                'toeflScore': toefl,
                'GRE_AWA': gre_awa,
                'GRE_QUANT': gre_quant,
                'GRE_VERBAL': gre_verbal,
                'GPA': gpa,
                'Publications': publication,
                'Work experience': work_exp,
                'Research Experience': re_exp,
                'Course Name Encoded': course,
                'scale': scale,
                'Degree': degree,
                'Budget': budget,
                'predictions':pred_unis
            })

        top_5_unis = []
        cnt = 0
        for prediction in predictions:
            if (cnt > 4):
                break
            top_5_unis.append(prediction[3])
            cnt = cnt + 1
        print("Top 5:", top_5_unis)
        df = pd.read_csv(unresampled_dataset)
        df_filtered = df[df['University Name'].isin(top_5_unis)]
        print(df_filtered.head())
        columns_to_keep = ['University Name', 'toeflScore', 'gre_total', 'GPA']
        df_filtered = df_filtered.loc[:, columns_to_keep]
        print(df_filtered)
        df_filtered['gre_total'] = df_filtered['gre_total'] + 260
        # Group by 'University' and calculate average for selected columns
        avg_scores_by_university = df_filtered.groupby('University Name').mean().reset_index()
        predictions1=predictions[:6]
        predictions2=predictions[6:12]
        predictions3=predictions[12:]
        print("predictions1",predictions1)
        print("predictions2",predictions2)
        print("predictions3",predictions3)

        return render_template('results.html', avg_scores_by_university=avg_scores_by_university,
                               predictions1=predictions1,predictions2=predictions2,predictions3=predictions3,admit_percent_overall=admit_percent_overall)
        # return render_template('results.html', avg_scores_by_university=avg_scores_by_university,
        #                        predictions=predictions,admit_percent_overall=(admit_percent_overall[0][0].round(4))*100)
        # return render_template('results.html',toefl_score_json=toefl_score_json ,gre_score_json=gre_score_json, gpa_json=gpa_json ,university_names_json=university_names_json ,
        #                        predictions=predictions)

    return render_template("recommendation_page.html", options=options, title="First page", form=form)


# for recommending course names

@app.route("/feedback", methods=['GET', 'POST'])
def feedback():
    options = []
    with open('Data Files/university_rankings.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            options.append((parts[1], parts[0]))  # (value, label)
    form = FeedbackForm()
    print("In feedback")
    return render_template('feedback.html', options=options, title="Feedback page", form=form)

def load_data():
    universities = {}
    courses = {}
    with open('Data Files\\university_courses (1).txt', 'r') as file:
        for line in file:
            parts = line.strip().split(' : ')
            if len(parts) == 2:
                university, course = parts
                if university in universities:
                    universities[university].append(course)
                else:
                    universities[university] = [course]
                courses[course] = university
    return universities, courses


@app.route("/rec_course",methods=['GET', 'POST'])
def view_course_page():
    # print("UserName:",g.user)
    options = []
    with open('Data Files/university_rankings.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            options.append((parts[1], parts[0]))  # (value, label)
    
    courseoptions=[]
    with open('Data Files/course_rankings.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            courseoptions.append(parts[0])  # (value, label)

    form = InputCourseForm()
    print(form.validate_on_submit())
    if form.validate_on_submit():
        print('in the form')
        gpa = form.gpa.data
        gre_awa = form.gre_awa.data
        gre_verbal = form.gre_verbal.data
        gre_quant = form.gre_quant.data
        toefl = form.toefl.data
        publication = form.publication.data
        re_exp = form.re_exp.data
        work_exp = form.work_exp.data
        course = form.course.data
        scale = form.scale.data
        degree=form.degree.data
        university_name=form.university.data

        course_idx=courseoptions.index(course)
        
        #preprocessing the input
        df=pd.read_csv(unresampled_dataset)
        re_df=pd.read_csv(resampled_dataset)
        # university_courses = df.groupby(university_name)['Course Name'].unique()
        numerical_columns = ['toeflScore', 'GRE_AWA', 'GRE_QUANT', 'GRE_VERBAL', 'GPA','min_gpa','gre_total']

        for col in numerical_columns:
            if col=='min_gpa' or col=='GPA' or col=='GRE_AWA':
                df[col]=df[col].astype(float).round(2)
            else:
                df[col]=df[col].astype(int)
        df = df.groupby('Course Name').filter(lambda x: len(x) >= 50)

        profile={
            'toeflScore':toefl,
            'GRE_AWA':gre_awa,
            'GRE_QUANT':gre_quant,
            'GRE_VERBAL':gre_verbal,
            'GPA': gpa,
            'Publications':publication,
            'Work experience':work_exp,
            'Research Experience':re_exp,
            'Course Name Encoded':course_idx,
            'scale':scale,
            'Degree Encoded': degree,
            'University Name Encoded': int(university_name)
        }
        print("University Name:",profile)
        #weights
        weights = {
            "toeflScore": 0.71,
            "GRE_QUANT": 0.85,
            "GRE_VERBAL": 0.99,
            "GPA": 0.65,
            "University Name Encoded": 0.96,
            "Degree Encoded": 0.90,
            "Work experience": 0.68
        }

        re_df['Score']=re_df['toeflScore']*weights['toeflScore']+re_df['Work experience']*weights['Work experience']+re_df['GRE_QUANT']*weights['GRE_QUANT']+re_df['GRE_VERBAL']*weights['GRE_VERBAL']+re_df['GPA']*weights['GPA']+re_df['University Name Encoded']*weights['University Name Encoded']+re_df['Degree Encoded']*weights['Degree Encoded']
        min_score = re_df['Score'].min()

        # Normalize the Score column
        re_df['Normalized Score'] = re_df['Score'] - min_score

        profile_df = pd.DataFrame([profile])
        # print(profile_df)

        # GRE scores processing
        profile_df['GRE_QUANT'] = profile_df['GRE_QUANT'] - 130
        profile_df['GRE_VERBAL'] = profile_df['GRE_VERBAL'] - 130

        # GPA conversion function
        def convert_gpa(gpa, scale):
            if scale == 10:
                return gpa / 2.5
            elif scale ==5:
                return (gpa/5)*4
            elif scale == 20:
                return gpa / 5.0
            elif scale == 100:
                return gpa / 25.0
            else:
                return gpa  

        
        # Convert GPA column to a 4.0 scale
        profile_df['GPA'] = profile_df.apply(lambda row: convert_gpa(row['GPA'], scale=scale), axis=1)

        #Concatenate
        df = pd.concat([df, profile_df], ignore_index=True)
        df.drop(['min_gpa','gre_total','min_gre'],axis=1,inplace=True)

        #Scaling the profile using RobertScalar
        scaler = RobustScaler()
        columns=['GPA','GRE_AWA','GRE_QUANT','GRE_VERBAL','toeflScore','Publications','Research Experience','Work experience']
        df[columns] = scaler.fit_transform(df[columns])
        ip_df = df.iloc[[-1]]
        # print(ip_df.info())

        #final preprocessed df for input
        input=ip_df[['GRE_QUANT', 'GRE_VERBAL', 'GPA',
                    'Work experience','Degree Encoded',
                    'University Name Encoded']]
        logits = courseModel.predict(input)  # Replace with actual prediction function

        # Extract probabilities and corresponding class labels
        probabilities = logits.ravel()  # Flatten the array
        class_labels = np.arange(len(probabilities))  # Assuming class indices start from 0

        # Get indices of top 10 predicted classes
        top_10_indices = np.argsort(probabilities)[::-1][:65]

        # Get corresponding probabilities and class labels for top 10
        top_10_probabilities = probabilities[top_10_indices]
        top_10_class_labels = class_labels[top_10_indices]

        # class to course
        def create_class_to_university_dict(file_path):
            class_to_course = {}
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        class_number = int(parts[1].strip())
                        course_name = parts[0].strip()
                        class_to_course[class_number] = course_name
            return class_to_course
        
        def create_class_to_university_dict(file_path):
            class_to_university = {}
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        class_number = int(parts[1].strip())
                        university_name = parts[0].strip()
                        class_to_university[class_number] = university_name
            return class_to_university

        class_to_university = create_class_to_university_dict('Data Files/university_rankings.txt')
        class_to_course = create_class_to_university_dict('Data Files/course_rankings.txt')
        predictions = []

        # Extract probabilities and corresponding class labels
        probabilities = logits.ravel()  # Flatten the array
        class_labels = np.arange(len(probabilities))  # Assuming class indices start from 0

        # Get indices of top 10 predicted classes
        top_10_indices = np.argsort(probabilities)[::-1][:65]

        # Get corresponding probabilities and class labels for top 10
        top_10_probabilities = probabilities[top_10_indices]
        top_10_class_labels = class_labels[top_10_indices]

        # Display top 10 predicted classes along with their probabilities
        predictions=[]
        for i in range(65):
            if top_10_class_labels[i]==int(profile['Course Name Encoded']):
                predictions.append([top_10_class_labels[i], top_10_probabilities[i]])
                filtered_df = re_df[re_df['University Name Encoded'] == profile['University Name Encoded']]
                max_score=filtered_df['toeflScore'].max()*weights['toeflScore']+filtered_df['Work experience'].max()*weights['Work experience']+filtered_df['GRE_QUANT'].max()*weights['GRE_QUANT']+filtered_df['GRE_VERBAL'].max()*weights['GRE_VERBAL']+filtered_df['GPA'].max()*weights['GPA']+weights['University Name Encoded']*117+weights['Degree Encoded']
            #     re_df['Score']=re_df['toeflScore']*weights['toeflScore']+re_df['Work experience']*weights['Work experience']+re_df['GRE_QUANT']*weights['GRE_QUANT']+re_df['GRE_VERBAL']*weights['GRE_VERBAL']+re_df['GPA']*weights['GPA']+re_df['University Name Encoded']*weights['University Name Encoded']+re_df['Degree Encoded']*weights['Degree Encoded']

                print(max_score)
                score=ip_df['Work experience']*weights['Work experience']+ip_df['GRE_QUANT']*weights['GRE_QUANT']+ip_df['GRE_VERBAL']*weights['GRE_VERBAL']+ip_df['GPA']*weights['GPA']+weights['University Name Encoded']*(116-ip_df['University Name Encoded'])+weights['Degree Encoded']
                admit_percent=((score-min_score)/(max_score-min_score))*100
                admit_percent=admit_percent.round(2)
                print("Class:", top_10_class_labels[i], " Probability:", top_10_probabilities[i], " Admit Percent:", admit_percent)

                # Map class number to university name
                course_name = class_to_course.get(top_10_class_labels[i])
                predictions[-1].append(course_name)  # Append university name to the last prediction
                predictions[-1].append(admit_percent.tolist()[0])  # Append university name to the last prediction
                predictions[-1].append(class_to_university.get(profile['University Name Encoded']))  # Append university name to the last prediction
                print("Course preds",predictions)

        return render_template('course_results.html', predictions=predictions)

    return render_template("course_recommendation_page.html", options=options,title="First page", form=form)

# @app.route('/chatbot', methods=['GET', 'POST'])
# def chat():
#     if request.method == 'POST':
#         print('POST')
#         message = request.form['userMessage']
#         print(message)
#         response = generate_response(message)
#         print(response)
#         return render_template('chatbot.html', response=response)
#     else:
#         # Handle GET request (if necessary)
#         # For example, render a page with the chatbot interface
#         return render_template('chatbot.html')

def translate_long_text(text):
    
    chunk_size = 500  # Maximum text length per request

    # Split the text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Translate each chunk and concatenate the translations
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    translated_text = ' '.join(translated_chunks)

    return translated_text

@app.route('/chatbot', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        message = request.form['userMessage']
        print(message)
        response = generate_response(message)
        print(response)
        lang = detect(response)
        if lang == 'zh-cn' or lang == 'zh-tw':
            response = translate_long_text(response)
        #     translated = translator.translate(response, src='zh-cn', dest='en')
        #     response = translated.text
        
        return jsonify(response=response)
    else:
        # Handle GET request (if necessary)
        # For example, render a page with the chatbot interface
        return render_template('chatbot.html')


def generate_response(message):
    # Use the completion endpoint of the ChatGPT API
    response = chatgpt_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}],
        # stream=True,
    )
    return response.choices[0].message.content


@app.route("/second")
def view_second_page():
    return render_template("homepage.html", title="Second page")


# login registration

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name=request.form['name']
        email=request.form['email']
        phone=request.form['phone']
        country=request.form['country']
        ugcollege=request.form['ugcollege']
        ugdegree=request.form['ugdegree']
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists. Choose a different one.', 'danger')
        else:
            users_collection.insert_one({'name':name,'email':email,'phone':phone,'country':country,'ugcollege':ugcollege,'ugdegree':ugdegree,'username': username, 'password': password})
            flash('Registration successful. You can now log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # g.user=username
        session['username'] = username
        
        # Check if the username and password match
        user = users_collection.find_one({'username': username, 'password': password})
        print("User",user)
        print("Username",username)
        print("Pass",password)
        if user is not None:
            flash('Login successful.', 'success')

            return redirect(url_for('view_home'))
            # Add any additional logic, such as session management
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            
        
       

    return render_template('login.html')


# validation 

@app.route('/admits_number', methods=['GET','POST'])
def getNumber():
    if request.method == 'POST':
                # name_attr='selectunis_'+str(i+1)
                # selected_uni=request.form[name_attr]
                print('entered post')
                number=request.form['number_of_unis']
                print("number in func:",number)
                return number
    else :
        return -1

import logging
logging.basicConfig(level=logging.WARNING)
@app.route('/admits', methods=['GET','POST'])
def valid():
    options = []
    with open('Data Files/university_rankings.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            if len(parts) == 2:  # Check if there are at least two parts after splitting
                options.append((parts[1], parts[0]))  # (value, label)
            else:
                # Log a warning for lines with insufficient parts
                logging.warning(f"Ignoring line in university_rankings.txt: {line.strip()}")
                # You may choose to skip the line, continue processing, or take other appropriate action


    actual_admits_arr=[]  
    # print("In valid")  
    number=getNumber()
    print("Number:",number)
    if request.method == 'POST':
            for i in range(int(number)):
                name_attr='selectunis_'+str(i+1)
                selected_uni=request.form[name_attr]
                def create_class_to_university_dict(file_path):
                    class_to_course = {}
                    with open(file_path, 'r') as file:
                        for line in file:
                            parts = line.strip().split(':')
                            if len(parts) == 2:
                                class_number = int(parts[1].strip())
                                course_name = parts[0].strip()
                                class_to_course[class_number] = course_name
                    return class_to_course

                class_to_university = create_class_to_university_dict('Data Files/university_rankings.txt')
                university_name = class_to_university.get(int(selected_uni))
                actual_admits_arr.append(university_name)
            print(actual_admits_arr)
            username = session.get('username')
            if username:
                actual_admits.insert_one({'username':username,'admits':actual_admits_arr})
            
    return render_template('admits.html',options=options)


if (__name__ == '__main__'):
    app.run()
