import streamlit as st
import pandas as pd
import joblib
import requests
import os

# Define the URL to your file in GitHub Releases
url = 'https://github.com/Tumup/ParisPrediction2024/releases/download/v1.0.0/voting_clf_model.pkl'

# Define the model directory (assuming it's the current directory)
model_dir = '.'

# Check if the file already exists
model_path = os.path.join(model_dir, 'voting_clf_model.pkl')
if not os.path.exists(model_path):
    st.write("Downloading model...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.write("Download completed.")
    except requests.exceptions.RequestException as e:
        st.write(f"Error downloading the model: {e}")

# Load the dataset to extract options for the dropdowns
data_path = os.path.join('..', 'DATA', 'final_filtered_athlete_games.csv')
st.write(f"Loading dataset from {data_path}")
df = pd.read_csv(data_path)

# Extract unique options for the dropdowns
sports = df['Sport'].unique().tolist()
nocs = df['NOC'].unique().tolist()
cities = df['City'].unique().tolist()
teams = df['Team'].unique().tolist()
events = df['Event'].unique().tolist()

# Load the model and encoders
try:
    st.write(f"Loading model from {model_path}")
    voting_clf = joblib.load(model_path)
    
    team_encoder_path = os.path.join(model_dir, 'team_encoder.pkl')
    st.write(f"Loading team encoder from {team_encoder_path}")
    team_encoder = joblib.load(team_encoder_path)
    
    event_encoder_path = os.path.join(model_dir, 'event_encoder.pkl')
    st.write(f"Loading event encoder from {event_encoder_path}")
    event_encoder = joblib.load(event_encoder_path)
    
    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    st.write(f"Loading label encoder from {label_encoder_path}")
    label_encoder = joblib.load(label_encoder_path)
    
    onehot_encoder_path = os.path.join(model_dir, 'onehot_encoder.pkl')
    st.write(f"Loading onehot encoder from {onehot_encoder_path}")
    onehot_encoder = joblib.load(onehot_encoder_path)
    
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    st.write(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    st.write(f"Loading feature names from {feature_names_path}")
    feature_names = joblib.load(feature_names_path)
    
except FileNotFoundError as e:
    st.write(f"Error loading model or encoders: {e}")
    st.stop()  # Stop the script if critical files are not found

st.title("Olympic Medal Prediction")

# User inputs
age = st.number_input("Age", min_value=12, max_value=70)
year = st.selectbox("Year", options=[2024])
sport = st.selectbox("Sport", options=sports)
season = st.selectbox("Season", options=["Summer"])
gender = st.selectbox("Gender", options=["M", "F"])
noc = st.selectbox("NOC", options=nocs)
city = st.selectbox("City", options=cities)
team = st.selectbox("Team", options=teams)
event = st.selectbox("Event", options=events)

if st.button("Predict"):
    data = {
        'Age': age,
        'Year': year,
        'Sport': sport,
        'Season': season,
        'Gender': gender,
        'NOC': noc,
        'City': city,
        'Team': team,
        'Event': event
    }
    df_input = pd.DataFrame([data])

    try:
        # Normalization (for 'Age' and 'Year')
        df_input[['Age', 'Year']] = scaler.transform(df_input[['Age', 'Year']])
        
        # One-Hot Encoding
        categorical_columns_onehot = ['Sport', 'Season', 'Gender', 'NOC', 'City']
        df_encoded = onehot_encoder.transform(df_input[categorical_columns_onehot])
        df_encoded = pd.DataFrame(df_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns_onehot))
        
        # Label Encoding
        df_input['Team'] = team_encoder.transform(df_input['Team'])
        df_input['Event'] = event_encoder.transform(df_input['Event'])

        # Combine encoded data
        df_final = pd.concat([df_input[['Age', 'Year']], df_encoded, df_input[['Team', 'Event']]], axis=1)
        
        # Ensure the order of columns matches what the model expects
        df_final = df_final.reindex(columns=feature_names)

        # Prediction
        y_prob = voting_clf.predict_proba(df_final)

        prob_df = pd.DataFrame(y_prob, columns=label_encoder.classes_)

        # Retrieve the team names
        team_index = df_input['Team'].values[0]
        team_name = team_encoder.inverse_transform([team_index])[0]

        st.write(f"Gold Medal Probability for {team_name}: {prob_df.loc[0, 'Gold']:.2f}")
        st.write(f"Silver Medal Probability for {team_name}: {prob_df.loc[0, 'Silver']:.2f}")
        st.write(f"Bronze Medal Probability for {team_name}: {prob_df.loc[0, 'Bronze']:.2f}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")
