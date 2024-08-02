
# Amazon Book Review Insight Project

## Project Overview
This project aims to enhance customer satisfaction and improve book recommendations on Amazon by analyzing user reviews. I utilized a dataset comprising user reviews of books, including review text and corresponding scores, to address these goals. This [Amazon Book Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data) dataset is well-suited for the problem as it provides direct feedback from customers about their reading experiences, enabling us to analyze sentiment and predict review scores.

## Table of Contents
1. [Business Understanding](#business-understanding)
2. [Data Understanding](#data-understanding)
3. [Data Preparation](#data-preparation)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Business Recommendations](#business-recommendations)
7. [Installation and Usage](#installation-and-usage)
8. [Contributing](#contributing)
9. [License](#license)

## Business Understanding
Amazon aims to enhance customer satisfaction and improve book recommendations. By leveraging user reviews, we can gain insights into customer sentiment and predict review scores, helping to tailor recommendations and identify areas for product improvement.

## Data Understanding
The data preparation focused on text preprocessing to ensure the review text was ready for modeling. Steps included removing HTML tags, converting text to lowercase, and tokenizing. I removed stopwords to eliminate common words that do not contribute to sentiment analysis and applied lemmatization to reduce words to their base forms. Libraries such as BeautifulSoup for HTML tag removal, nltk for tokenization, stopword removal, and lemmatization, and sklearn for feature extraction using TF-IDF were used. These steps ensured that the textual data was clean, standardized, and suitable for feature extraction and modeling.

I also employed several machine learning models using scikit-learn and xgboost libraries. The models included Logistic Regression, Random Forest, and XGBoost.

This dataset provides direct feedback from customers about their reading experiences, making it well-suited for sentiment analysis and predictive modeling.

## Data Preparation
### Text Cleaning
To prepare the data for analysis and modeling, we perform the following text cleaning steps:



### Feature Extraction
We use the TF-IDF (Term Frequency-Inverse Document Frequency) method to convert the cleaned text data into numerical features suitable for machine learning models.

## Modeling
We implemented several machine learning models using `scikit-learn` and `xgboost` libraries:

### Logistic Regression
1. **Model Selection**: Chosen for its simplicity and interpretability.
2. **Hyperparameter Tuning**: Used GridSearchCV to optimize parameters.
3. **Training**: Trained the model on the training set with optimal hyperparameters.
[lr3]('images/lr3_corr')
### Random Forest
1. **Model Selection**: Chosen for its robustness and feature importance evaluation.
2. **Hyperparameter Tuning**: Used RandomizedSearchCV to optimize parameters.
3. **Training**: Trained the model on the training set with optimal hyperparameters.

### XGBoost
1. **Model Selection**: Chosen for its high performance and ability to handle imbalanced datasets.
2. **Hyperparameter Tuning**: Used RandomizedSearchCV to optimize parameters.
3. **Training**: Trained the model on the training set with optimal hyperparameters.

## Evaluation
The models were evaluated using accuracy, precision, recall, and F1-score. I also used confusion matrices to visualize the performance of each model.

### Model Performance
- **Logistic Regression**: Accuracy: 0.79, Precision: 0.76, Recall: 0.69, F1-Score: 0.72
- **Random Forest**: Accuracy: 0.76, Precision: 0.68, Recall: 0.71, F1-Score: 0.69
- **XGBoost**: Accuracy: 0.76, Precision: 0.66, Recall: 0.81, F1-Score: 0.73

### Interpretation of the scores

__Accuracy:__ In this project, accuracy tells us the overall correctness of our model predictions. However, because we might have an imbalance between positive and negative reviews, accuracy alone isn't enough to judge the model performance.

__Precision:__ Precision is crucial in this context because it indicates the quality of the positive review predictions. High precision means that when the model predicts a review as positive, it is likely correct. This is important for recommending books because false positives (recommending a book that isn't actually well-liked) can harm customer trust.

__Recall:__ Recall is important because it shows the model's ability to capture actual positive reviews. High recall means fewer missed opportunities for recommending good books. For Amazon, missing out on recommending good books (false negatives) means lost sales opportunities.

__F1-Score:__ The F1-Score balances precision and recall, providing a single metric that accounts for both false positives and false negatives. In the context of book recommendations, a high F1-Score means that the model maintains a good balance between recommending books that users will like and not missing out on recommending good books.

### Sentiment Analysis
I performed sentiment analysis using `TextBlob` and `VADER`:

__TextBlob:__ The sentiment scores provided by TextBlob are easy to interpret, which is useful for gaining quick insights into the data.

__VADER:__ VADER is specifically designed for sentiment analysis of social media text. It handles informal language, slang, emojis, and other nuances often found in user-generated content, making it well-suited for analyzing Amazon book reviews.

- **TextBlob Sentiment (Binary)**: Accuracy: 0.43, Precision: 0.39, Recall: 0.81, F1-Score: 0.52
- **VADER Sentiment (Binary)**: Accuracy: 0.45, Precision: 0.39, Recall: 0.71, F1-Score: 0.50

### Interpretation in Terms of the Project
__Accuracy:__ In the context of sentiment analysis, accuracy indicates the overall correctness of the sentiment predictions. Both TextBlob and VADER show low accuracy, suggesting that they may not be reliable on their own for classifying sentiments in reviews.

__Precision:__ Precision is critical in sentiment analysis to avoid recommending books that are actually negatively reviewed (false positives). Both TextBlob and VADER have low precision (0.39), indicating a high rate of false positives, which is problematic for maintaining customer trust.

__Recall:__ Recall is important for capturing as many positive reviews as possible. TextBlob has a higher recall (0.81) compared to VADER (0.71), indicating that TextBlob is better at identifying positive sentiments but at the cost of higher false positives.

__F1-Score:__ The F1-Score balances precision and recall. TextBlob (0.52) and VADER (0.50) both have low F1-Scores, reflecting their overall struggle to accurately and reliably classify sentiments.

## Business Recommendations
Based on the model performance and sentiment analysis, we recommend:
1. __Improve Product Recommendations__

    - __What to Do:__ Use the Logistic Regression model to predict how likely customers are to rate a product positively based on their reviews.

    - __Why:__ This model provides reliable predictions, helping to recommend products that customers are more likely to love, enhancing their shopping experience.


2.  __Target Satisfied Customers for Testimonials and Promotions__

    -  __What to Do:__ Utilize the XGBoost model to identify highly satisfied customers who can provide testimonials or be targeted for promotional campaigns.
    
    - __Why:__ This model is excellent at identifying happy customers, allowing Amazon to leverage positive feedback for marketing purposes and build a stronger brand reputation.  

3. __Identify and Address Common Customer Issues__

    - __What to Do:__ Use sentiment analysis tools like TextBlob and VADER to analyze customer reviews for common issues or complaints.
    
    - __Why:__ Understanding the common pain points in customer reviews helps in making targeted improvements to products and services, leading to increased customer satisfaction and loyalty. 

## Next Steps

1. __Invest in Advanced Text Analysis Tools__

    - __What to Do:__ Explore advanced Natural Language Processing (NLP) models like BERT to further improve sentiment analysis accuracy.
    
    - __Why:__ More sophisticated models can provide deeper insights into customer sentiments, leading to better decision-making and more precise targeting of customer needs.
2. __Continuous Model Improvement__

    - __What to Do:__ Regularly update and fine-tune the predictive models with new data to maintain their accuracy and effectiveness.
    
    - __Why:__ Keeping models up-to-date ensures they continue to provide accurate predictions, adapting to changing customer preferences and behaviors. 
