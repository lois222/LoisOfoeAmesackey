# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd

# Loading the trained machine learning model and scaler from the disk
rf_model = joblib.load("C:/Users/user/OneDrive - Ashesi University/Desktop/Regression Assignment/Fifaprediction.pkl")
scaler = joblib.load("C:/Users/user/OneDrive - Ashesi University/Desktop/Regression Assignment/Scaler.pkl")

# Function to preprocess user input
def preprocess_input(value_eur,age, potential, movement_reactions, club_joined_date ):
    club_joined_date = float(club_joined_date.replace('-', ''))

    # DataFrame with user input
    user_input = pd.DataFrame({
        'value_eur': [value_eur],
        'age': [age],
        'potential': [potential],
        'movement_reactions': [movement_reactions],  
        'club_joined_date': [club_joined_date]
        
    })

    # Scaling the user input
    user_input_scaled = scaler.transform(user_input)  

    return user_input_scaled

# Main function for Streamlit app
def main():
    # Seting the Streamlit page configuration
    st.set_page_config(
        page_title='⚽ FIFA Player Prediction',
        page_icon='⚽',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.markdown(
        """
        <style>
            body {
                background-color: #f1f1f1;
            }
            .sidebar .sidebar-content {
                background-color: #2C3E50;
                color: #ECF0F1;
            }
            .sidebar .widget-content {
                background-color: #34495E;
            }
            .streamlit-button {
                background-color: #3498DB;
                color: #ECF0F1;
            }
            .streamlit-button:hover {
                background-color: #2980B9;
            }
            .stSlider>div>div>div>div {
                background-color: #3498DB;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and welcome message
    st.title('⚽ FIFA Player Overall Rating Prediction ⚽')
    st.write(
        """
        Welcome to the FIFA Player Overall Rating Prediction app! 🎉
        Select the input features on the left, and we'll predict the player's overall rating.
        """
    )

    # Input features in the sidebar
    st.sidebar.header('Input Features')

    # Sidebar sliders and input fields for user input
    value_eur = st.sidebar.slider('Value (in Euros)', 0.0, 100000000.0, 50.0)
    age = st.sidebar.slider('Age', 18, 50, 20)
    potential = st.sidebar.slider('Potential', 50, 100, 75)
    movement_reactions = st.sidebar.slider('movement_reactions', 0, 100, 10)
    club_joined_date = st.sidebar.text_input('Club_Joined_date', '1990-01-01')

    if st.sidebar.button('Predict'):
        # Preprocessing the  user input and making predictions
        user_input_scaled = preprocess_input(value_eur, age, potential, movement_reactions, club_joined_date)
        prediction = rf_model.predict(user_input_scaled)

        # Displaying the prediction result
        st.subheader('Prediction')
        st.write(f'The predicted overall rating is: {prediction[0]}')

# Running the Streamlit app
if __name__ == '__main__':
    main()

# streamlit run deployFifa.py (Run in command line to run app)
