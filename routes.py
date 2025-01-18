from nltk.stem import PorterStemmer
from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import pandas as pd
import os
import numpy as np
import pickle
from gensim.models.fasttext import FastText
import difflib


# Initialize NLTK's PorterStemmer
ps = PorterStemmer()

main = Blueprint('main', __name__)

# Load datasets and models
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'cloth_data.csv'))

# Get the current directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model file
models_path = os.path.join(base_dir, 'models')

# Add 'id' column to use as item identifiers
df['id'] = df.index

# Load FastText model
fasttext_model = FastText.load(os.path.join(models_path, "fasttext_model.bin"))

# Load Logistic Regression model
with open(os.path.join(models_path, "logistic_regression_model.pkl"), 'rb') as f:
    logistic_regression_model = pickle.load(f)

# Load TF-IDF Vectorizer
with open(os.path.join(models_path, "tfidf_vectorizer.pkl"), 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def tokenize_and_process_review(text):
    words = text.lower().split()
    return [ps.stem(word) for word in words if word in fasttext_model.wv]  # Stemming is applied here

def vectorize_review(text):
    tokens = tokenize_and_process_review(text)
    if tokens:
        return np.mean([fasttext_model.wv[token] for token in tokens], axis=0)
    else:
        return np.zeros(fasttext_model.vector_size)

# Function to use FastText for search queries
def fasttext_search(query, df):
    search_terms = query.lower().split()
    search_terms = [ps.stem(term) for term in search_terms]  # Applied stemming to the search terms
    query_vector = np.mean([fasttext_model.wv[term] for term in search_terms if term in fasttext_model.wv], axis=0)
    
    matched_items = []

    for _, row in df.iterrows():
        review_vector = vectorize_review(row['Review Text'])
        if np.linalg.norm(query_vector) > 0 and np.linalg.norm(review_vector) > 0:
            similarity = np.dot(query_vector, review_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(review_vector))
            if similarity > 0.5:
                matched_items.append(row)

    return matched_items

# Fuzzy search using difflib 
def fuzzy_search(query, df, threshold=0.6):
    search_terms = query.lower().split()
    search_terms = [ps.stem(term) for term in search_terms]  # Applied stemming here too
    matched_items = []

    for _, row in df.iterrows():
        review_text = row['Review Text'].lower()
        title = row.get('Title', '').lower()  # To check  if 'Title' exists in the dataframe

        for term in search_terms:
            if is_fuzzy_match(term, review_text, threshold) or is_fuzzy_match(term, title, threshold):
                matched_items.append(row)
                break  # If one match is found, add the item and stop checking further terms

    return matched_items

# Helper function to use difflib for fuzzy matching
def is_fuzzy_match(term, text, threshold=0.6):
    matches = difflib.get_close_matches(term, [text], n=1, cutoff=threshold)
    return len(matches) > 0

def getProductByID(id):
    # Check if there are any rows that match the Clothing ID
    matching_row = df.loc[df["Clothing ID"] == id]
    
    # If the matching_row DataFrame is not empty, return the desired columns
    if not matching_row.empty:
        df_1 = matching_row[["Clothing ID", "Clothes Description", "Rating", "Clothes Title", "Class Name"]].iloc[0]  # Using iloc[0] to get the first matching row
        # print(df_1)
        return df_1

    return None

# Function to vectorize the review using TF-IDF
def vectorize_review_with_tfidf(text, tfidf_vectorizer):
    # Transform the text using the TF-IDF vectorizer
    return tfidf_vectorizer.transform([text]).toarray()

# Homepage route
@main.route('/')
def home():
    return render_template('home.html')

# Search page route using FastText
@main.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return redirect(url_for('main.home')) 

    # Call the new search and process results function to avoid duplicates
    processed_results = search_and_process_results(query, df)
    
    # Convert processed_results to a list of dictionaries
    processed_results = processed_results.to_dict(orient='records')  
    
    #print(processed_results)

    return render_template('search_results.html', query=query, results=processed_results)

#  route for fuzzy search using difflib
@main.route('/fuzzy_search', methods=['GET'])
def fuzzy_search_route():
    query = request.args.get('query')
    if not query:
        return redirect(url_for('main.home'))  # Redirect to homepage if no query

    # fuzzy_search function
    fuzzy_results = fuzzy_search(query, df)

    return render_template('search_results.html', query=query, fuzzy_results=fuzzy_results)

# Product detail page route
@main.route('/product/<int:product_id>', methods=['GET'])
def product_detail(product_id):
    product = getProductByID(product_id) 
# Get the recommendation status from the query parameters, default to None
    recommends = request.args.get('recommends', default=None, type=lambda v: v.lower() == 'true')
    # Fetch reviews (this is your logic to get the list of reviews)
    reviews = get_reviews_by_product_id(product_id, "most_helpful", 10)

    return render_template('product_detail.html', product=product, reviews=reviews, recommends=recommends)


CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'cloth_data.csv')

@main.route('/submit_review/<int:product_id>', methods=['POST'])
def submit_review(product_id):
    review_title = request.form['reviewTitle']
    review_rating = request.form['reviewRating']
    review_text = request.form['reviewText']
    review_age = request.form['reviewAge']

    # Process the review to get the recommendation status
    recommends_product = process_review(review_text)

    df = pd.read_csv(CSV_FILE_PATH)

    # Check if the product_id exists in the DataFrame to fill in missing columns
    existing_product = df[df['Clothing ID'] == product_id]

    if not existing_product.empty:
        department_name = existing_product['Department Name'].values[0]
        class_name = existing_product['Class Name'].values[0]
        clothes_title = existing_product['Clothes Title'].values[0]
        clothes_description = existing_product['Clothes Description'].values[0]
        division_name = existing_product['Division Name'].values[0]
    else:
        # Default values in case product_id does not exist in the CSV
        department_name = 'Unknown'
        class_name = 'Unknown'
        clothes_title = 'Unknown'
        clothes_description = 'No description available'

    # Prepare the new review data as a dictionary
    new_review_data = {
        'Clothing ID': product_id,
        'Age': review_age,
        'Title': review_title,
        'Review Text': review_text,
        'Rating': review_rating,
        'Recommended IND': recommends_product,
        'Positive Feedback Count': 0,  # Assuming a default value of 0 for positive feedback count
        'Division Name': division_name,
        'Department Name': department_name,
        'Class Name': class_name,
        'Clothes Title': clothes_title,
        'Clothes Description': clothes_description
    }

    # Convert the new review data to a DataFrame
    new_review = pd.DataFrame([new_review_data])

    df = pd.concat([df, new_review], ignore_index=True)

    df.to_csv(CSV_FILE_PATH, index=False)

    return redirect(url_for('main.product_detail', product_id=product_id, reviewTitle=review_title, reviewRating=review_rating, reviewText=review_text, reviewAge=review_age, recommends=recommends_product))



@main.route('/save_review/<int:product_id>', methods=['POST'])
def save_review(product_id):
    # Retrieve all the review data and the final recommendation outcome
    review_title = request.form['reviewTitle']
    review_rating = request.form['reviewRating']
    review_text = request.form['reviewText']
    review_age = request.form['reviewAge']
    recommendation = request.form['recommendation']

    # Save the review to the database 
    save_review_to_db(product_id, review_title, review_rating, review_text, review_age, recommendation) 

    # Redirect back to the product detail page after saving
    return redirect(url_for('main.product_detail', product_id=product_id))

# Function to vectorize the entered review text
def vectorize_review_text(text, fasttext_model, tfidf_vectorizer):
    words = text.split()  # Split the text into words
    tfidf_weights = tfidf_vectorizer.transform([text]).toarray()[0]  
    feature_names = tfidf_vectorizer.get_feature_names_out()  
    weighted_vectors = []

    # For each word in the review, if it's in the FastText model and TF-IDF vocabulary, calculate its weighted vector
    for word in words:
        if word in feature_names:
            try:
                weighted_vectors.append(fasttext_model.wv[word] * tfidf_weights[feature_names.tolist().index(word)])
            except KeyError:
                pass

    # If there are weighted vectors, return their average; otherwise, return a zero vector
    if len(weighted_vectors) > 0:
        return sum(weighted_vectors) / len(weighted_vectors)
    else:
        return np.zeros(fasttext_model.vector_size)


# Function to process the review using the Logistic Regression model
def process_review(review_text):
    # Vectorize the review using TF-IDF vectorizer
    review_vector = vectorize_review_with_tfidf(review_text, tfidf_vectorizer)

    # Use the Logistic Regression model to predict the sentiment 
    prediction = logistic_regression_model.predict(review_vector)

    return prediction[0] == 1

def save_review_to_db(product_id, review_title, review_rating, review_text, review_age, recommendation):
    print(f"Saving review for product {product_id}:")
    print(f"Title: {review_title}, Rating: {review_rating}, Text: {review_text}, Age: {review_age}, Recommendation: {recommendation}")  


@main.route('/categories', methods=['GET'])
def categories():
    """Display a list of unique divisions."""
    divisions = df['Division Name'].dropna().unique().tolist()
    return render_template('categories.html', divisions=divisions)

# Route to display departments within a division
@main.route('/browse/<division_name>/departments', methods=['GET'])
def browse_departments(division_name):
    """Display departments under the selected division."""
    departments = df[df['Division Name'] == division_name]['Department Name'].dropna().unique().tolist()
    return render_template('categories.html', departments=departments, current_division=division_name)

# Route to display class names within a department
@main.route('/browse/<department_name>/classes', methods=['GET'])
def browse_classes(department_name):
    """Display class names under the selected department."""
    class_names = df[df['Department Name'] == department_name]['Class Name'].dropna().unique().tolist()
    return render_template('categories.html', class_names=class_names, current_department=department_name)

# Route to display products within a class
@main.route('/browse/<class_name>/products', methods=['GET'])
def browse_clothes(class_name):
    """Display products under the selected class."""
    filtered_products = df[df['Class Name'] == class_name]
    products = filtered_products.to_dict(orient='records')
    return render_template('browse.html', class_name=class_name, products=products)

@main.route('/reviews/<int:product_id>', methods=['GET'])
def reviews(product_id):
    product = getProductByID(product_id)

    return render_template('reviews.html', product=product)

@main.route('/fetch_reviews/<int:product_id>', methods=['GET'])
def fetch_reviews(product_id):
    sort_by = request.args.get('sort_by', 'newest')

    df = pd.read_csv(CSV_FILE_PATH)

    # Filter reviews by the product ID
    reviews = df[df['Clothing ID'] == product_id]

    # Sort reviews based on the requested sorting method
    if sort_by == 'most_helpful':
        reviews = reviews.sort_values(by='Positive Feedback Count', ascending=False)
    elif sort_by == 'newest':
        # Sort by index, so the newest review 
        reviews = reviews.reset_index(drop=True).sort_index(ascending=False)

    # Convert the sorted reviews to a list of dictionaries
    reviews_list = reviews.to_dict(orient='records')

    # Return the reviews as JSON
    return jsonify({'reviews': reviews_list})



def get_reviews_by_product_id(product_id, sort_by, limit=None):
    # Filter the DataFrame by product ID
    reviews = df[df['Clothing ID'] == product_id]

    # Check if any reviews are found for this product
    if reviews.empty:
        return []

    # Select the relevant columns for the reviews
    reviews = reviews[['Title', 'Review Text', 'Rating', 'Positive Feedback Count']]

    # Sort the DataFrame based on the selected option
    if sort_by == 'most_helpful':
        reviews = reviews.sort_values(by='Positive Feedback Count', ascending=False)
    elif sort_by == 'newest':
        reviews = reviews.sort_index(ascending=False)

    reviews_list = reviews.to_dict(orient='records')

    if limit:
        reviews_list = reviews_list[:limit]

    return reviews_list



# function to search and process results, removing duplicates based on Clothing ID
def search_and_process_results(query_string, df):
    raw_results = fasttext_search(query_string, df)
    grouped_results = {}

    # Iterate over the search results
    for result in raw_results:
        clothing_id = result["Clothing ID"]

        # If the product already exists, update based on criteria 
        if clothing_id in grouped_results:
            existing_review = grouped_results[clothing_id]

            # Keep the review with the highest rating (or
            if result["Rating"] > existing_review["Rating"]:
                grouped_results[clothing_id] = result
        else:
            grouped_results[clothing_id] = result

    # Convert the grouped dictionary to a DataFrame 
    processed_results = pd.DataFrame.from_dict(grouped_results, orient='index')

    # Return the unique product entries
    return processed_results.reset_index(drop=True)


"""
References:

Flask Framework: Pallets Projects. (2023). Flask (Version 2.2.5) [Software]. https://flask.palletsprojects.com/

Pandas Library: The pandas development team. (2023). pandas-dev/pandas: Pandas (Version 1.5.3) [Software]. Zenodo. https://doi.org/10.5281/zenodo.3509134

NumPy Library: Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

NLTK Library: Loper, E., & Bird, S. (2002). NLTK: The natural language toolkit. Proceedings of the ACL-02 Workshop on Effective Tools and Methodologies for Teaching Natural Language Processing and Computational Linguistics, 63–70. https://www.nltk.org/

Gensim Library: Řehůřek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, 45–50. https://radimrehurek.com/gensim/

Scikit-Learn Library (for Logistic Regression): Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(85), 2825–2830. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

FastText Model: Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5, 135–146. https://doi.org/10.1162/tacl_a_00051

Difflib Module (Python Standard Library): Python Software Foundation. (2023). Difflib – Helpers for computing deltas [Software]. https://docs.python.org/3/library/difflib.html

Bootstrap Framework: Bootstrap. (2023). Bootstrap (Version 5.3) [Software]. https://getbootstrap.com/
"""