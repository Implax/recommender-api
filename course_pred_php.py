from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pymysql.cursors
import json

print("Working")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*", "headers": "*"}})


@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():


    selected_courses = request.json['selectedCourses']

    course_names = []
    course_majors = []

    for course in selected_courses:
        course_names.append(course['courseName'])
        course_majors.append(course['courseMajor'])

    # create a connection to the database

    connection = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='',
    database='ashesi_apps',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
    # execute a query to retrieve the course data
    with connection.cursor() as cursor:
        # Use the `IN` operator to check if the course_major is in the course_majors array
        sql = "SELECT course_name, course_description FROM courseplanner_electives WHERE course_major IN %s"
        cursor.execute(sql, [course_majors])

        # Fetch the results
        result = cursor.fetchall()

    # create a pandas DataFrame from the query result
    courses_df = pd.DataFrame(result, columns=['course_name', 'course_description'])

    # create a TF-IDF vectorizer object
    tfidf = TfidfVectorizer(stop_words='english')

    # fit and transform the course descriptions
    tfidf_matrix = tfidf.fit_transform(courses_df['course_description'])

    # create an LSA model to reduce the dimensionality of the vector space
    lsa = TruncatedSVD(n_components=20, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # calculate the cosine similarity matrix
    cos_sim = cosine_similarity(lsa_matrix)

    # get the selected courses from the request
    print(course_names)

    # get the index of the selected courses
    selected_indices = []
    for course in course_names:
        index = courses_df[courses_df['course_name'] == course].index[0]
        selected_indices.append(index)

    # calculate the average cosine similarity scores for the selected courses
    cos_sim_scores = cos_sim[selected_indices].mean(axis=0)

    # remove the indices of the selected courses from the array
    for index in selected_indices:
        cos_sim_scores[index] = 0.0

    # filter out courses with 0% similarity and get the top k courses with the highest cosine similarity
    top_k = 3
    nonzero_indices = cos_sim_scores.nonzero()[0]
    top_courses_indices = nonzero_indices[cos_sim_scores[nonzero_indices].argsort()[-top_k:][::-1]]
    top_courses = courses_df.iloc[top_courses_indices]['course_name'].tolist()
    top_cos_sim_scores = cos_sim_scores[top_courses_indices]

    # close the database connection
    connection.close()

    result = []

    for course, score in zip(top_courses, top_cos_sim_scores):
        result.append(f'{course}: {score*100:.2f}%')
    
    print(result)

    response_data = {"results": result}
    response = jsonify(response_data)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == '__main__':
    app.run(debug=True)
