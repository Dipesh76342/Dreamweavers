from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('uni\college.csv')

# Combine relevant features
df['Combined'] = df['City'] + ' ' + df['Subjects'] + ' ' + df['Country']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Combined'])

# Function to recommend universities based on user input
def recommend_universities(user_city, user_cost, user_subjects, user_ielts_score, user_country):
    user_input_combined = user_city + ' ' + user_subjects + ' ' + user_country
    user_vector = vectorizer.transform([user_input_combined])
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix)
    df['Similarity'] = cosine_sim[0]
    df_filtered = df[(df['City'].str.contains(user_city, case=False)) & 
                     (df['Country'].str.contains(user_country, case=False)) &
                     (df['Cost'] <= user_cost) & 
                     (df['IELTS Score'] <= user_ielts_score)]
    recommended_universities = df_filtered.sort_values(by='Similarity', ascending=False)
    return recommended_universities[['University', 'Cost', 'City', 'Country', 'IELTS Score']]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_city = request.form['city']
        user_country = request.form['country']
        user_cost = int(request.form['cost'])
        user_subjects = request.form['subjects']
        user_ielts_score = float(request.form['ielts'])
        
        recommendations = recommend_universities(user_city, user_cost, user_subjects, user_ielts_score, user_country)
        return render_template('results.html', recommendations=recommendations)
    return render_template('index.html')

@app.route('/university/<university_name>')
def university_detail(university_name):
    university = df[df['University'].str.lower() == university_name.lower()]
    if not university.empty:
        university = university.iloc[0]
        return render_template('university_detail.html', university_name=university_name, university=university)
    return "University not found", 404
if __name__ == '__main__':
    app.run(debug=True)
