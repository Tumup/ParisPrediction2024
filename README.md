# ParisPrediction2024

## Data Preparation

First, we have correctly encoded our dataset in UTF-8. Next, we filtered our data to include only the competitors who have won a medal, whether bronze, silver, or gold, over the years in the Summer Olympic Games. This was done in the file [`data_prep.ipynb`](./ONLY_MEDAL_WINNERS/data_prep.ipynb).


## Exploratory Data Analysis (EDA) Summary

All this information can be found in the file  [`EDA.ipynb`](./ONLY_MEDAL_WINNERS/EDA.ipynb)

 Exploratory Data Analysis (EDA), we performed the following steps:

1. **Initial Data Overview**:
   - Loaded the dataset and provided a general overview.
   - Identified that the only missing values were in the `Age` variable.

2. **Data Cleaning**:
   - Removed rows with missing values in the `Age` variable as they corresponded to very old Olympic participations.
   - Converted the `Age` variable from float to int for consistency.

3. **Outlier Detection**:
   - Verified that there were no outliers in the `Year` and `Entry ID` variables.
   - Identified outliers in the `Age` variable and decided to remove participants younger than 13 years old and older than 70 years old.

4. **Data Visualization**:
   - Created various visualizations to understand the data distribution:
     - A pie chart to show the gender distribution of the participants.
     - A horizontal bar chart to display the top 10 cities hosting Olympic events.
     - A horizontal bar chart for the distribution of sports in the dataset.
     - Horizontal bar charts for the top 10 countries with the most gold, silver, and bronze medals won.

5. **Correlation Matrix**:
   - Conducted a correlation matrix analysis for the numeric variables to understand the relationships between them.

6. **Final Data Preparation**:
   - Saved the cleaned and processed data for further analysis.

This comprehensive EDA allowed us to clean the dataset, identify key patterns and trends, and prepare the data for subsequent analysis.



# Ensemble Modeling for Predicting Olympic Medals
 
 All this information can be found in the file [`model.ipynb`](./ONLY_MEDAL_WINNERS/model.ipynb)

## Introduction

In this project, we aimed to predict the probability of winning medals for different teams in various events using an ensemble of machine learning models. The steps taken include data preprocessing, model training, and ensemble creation using `VotingClassifier`.

## Steps and Methodology

1. **Data Loading and Preprocessing:**
   - The dataset was loaded from a CSV file.
   - Unnecessary columns were dropped.
   - Data was split into training and test sets based on specific years (2012, 2016, 2020).

2. **Normalization:**
   - Applied `StandardScaler` to normalize numerical columns ('Age' and 'Year').

3. **One-Hot Encoding:**
   - Categorical columns ('Sport', 'Season', 'Gender', 'NOC', 'City') were one-hot encoded.

4. **Label Encoding:**
   - Encoded 'Team' and 'Event' columns using `LabelEncoder`.
   - Encoded the target variable 'Medal'.

5. **Model Selection and Training:**
   - Chose a variety of machine learning models including:
     - RandomForestClassifier
     - GradientBoostingClassifier
     - SVC (Support Vector Classifier)
     - XGBClassifier (Extreme Gradient Boosting)
     - AdaBoostClassifier
     - ExtraTreesClassifier
     - BaggingClassifier
     - LogisticRegression
     - GaussianNB (Gaussian Naive Bayes)
   - Used default or simplified parameters to reduce training time.

6. **Ensemble Creation:**
   - Combined the trained models into an ensemble using `VotingClassifier` with soft voting.

7. **Model Evaluation:**
   - Evaluated the ensemble on validation and test sets using:
     - Classification Report
     - Confusion Matrix
     - Accuracy Score
   - Predicted probabilities for each class (medal type).

8. **Result Presentation:**
   - Created a DataFrame to show the probabilities of winning gold, silver, and bronze medals for each event.
   - Displayed the final results.

## Conclusion

By using an ensemble of various machine learning models, we were able to create a robust predictive model for predicting Olympic medal winners. The use of `VotingClassifier` allowed us to leverage the strengths of multiple models, resulting in improved performance and accuracy. This approach demonstrates the power of ensemble learning in complex classification tasks.


# Olympic Medal Prediction App

 All this information can be found in the file [`app.py`](./STREAMLIT/app.py)

This Streamlit application predicts the probability of winning gold, silver, and bronze medals for athletes in the 2024 Olympics based on various features such as age, sport, gender, and more.

## Features

- **User Inputs:**
  - Age
  - Year (fixed to 2024)
  - Sport (dropdown with all sports from the dataset)
  - Season (fixed to Summer)
  - Gender (dropdown with options "M" and "F")
  - NOC (National Olympic Committee, dropdown with all options from the dataset)
  - City (dropdown with all options from the dataset)
  - Team (dropdown with all options from the dataset)
  - Event (dropdown with all options from the dataset)

- **Model Prediction:**
  - Predicts the probability of winning gold, silver, and bronze medals.
  - Displays the predicted probabilities for the selected team.
