import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
# Sets the title and icon of the browser tab
st.set_page_config(
    page_title="Fish Weight Prediction App",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- App Title and Description ---
st.title("🐟 Fish Weight Prediction App")
st.write(
    "This app predicts the weight of a fish based on its physical measurements. "
    "Provide the details of the fish in the sidebar to get a prediction."
)
st.write("---")

# --- Load The Trained Model ---
# We use a try-except block to gracefully handle errors if the model file is not found.
try:
    # The path should point to the .pkl file you downloaded from Colab
    model = joblib.load('fish_weight_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'fish_weight_model.pkl' is in the same directory as this script.")
    model = None # Set model to None so the app doesn't crash
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    model = None

# --- Sidebar for User Inputs ---
# st.sidebar creates a panel on the left where users can input data.
st.sidebar.header("Input Fish Features")

def user_input_features():
    """
    Creates sidebar widgets to get fish measurement inputs from the user.
    Returns the inputs as a NumPy array for the model.
    """
    # The list of species should match the ones in the original dataset.
    species_options = ['Perch', 'Bream', 'Roach', 'Pike', 'Smelt', 'Parkki', 'Whitefish']
    species = st.sidebar.selectbox('Species', sorted(species_options))

    # Sliders for numerical inputs. We set reasonable min, max, and default values.
    length1 = st.sidebar.slider('Vertical Length (cm)', 7.0, 60.0, 24.0)
    length2 = st.sidebar.slider('Diagonal Length (cm)', 8.0, 64.0, 26.0)
    length3 = st.sidebar.slider('Cross Length (cm)', 8.5, 69.0, 29.0)
    height = st.sidebar.slider('Height (cm)', 1.5, 19.0, 9.0)
    width = st.sidebar.slider('Width (cm)', 1.0, 8.5, 4.0)

    # Prepare data for the model
    # The model was trained on one-hot encoded species. We must replicate that here.
    # Create a dictionary for the species with the selected one set to 1.
    species_data = {f'Species_{s}': 0 for s in species_options}
    species_data[f'Species_{species}'] = 1

    # Create a dictionary for all input data
    data = {
        'Length1': length1,
        'Length2': length2,
        'Length3': length3,
        'Height': height,
        'Width': width,
        **species_data # Merge the species dictionary
    }

    # IMPORTANT: The order of columns in this DataFrame must EXACTLY match the
    # order of columns your model was trained on in the Colab notebook.
    # You may need to adjust this list based on your specific model.
    feature_order = [
        'Length1', 'Length2', 'Length3', 'Height', 'Width',
        'Species_Bream', 'Species_Parkki', 'Species_Perch', 'Species_Pike',
        'Species_Roach', 'Species_Smelt', 'Species_Whitefish'
    ]

    # Create a DataFrame and reorder columns to match the training order
    features_df = pd.DataFrame(data, index=[0])
    # Add any missing species columns and fill with 0
    for col in feature_order:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[feature_order]

    return features_df

# Get user input
input_df = user_input_features()

# --- Main Panel: Display Inputs and Prediction ---
st.header("Your Input Features")
st.write("The table below shows the features you have selected.")
st.table(input_df)
st.write("---")

# Prediction logic
# Only show the button and prediction if the model was loaded successfully.
if model is not None:
    if st.button('Predict Fish Weight', key='predict_button'):
        try:
            prediction = model.predict(input_df)
            st.subheader("Predicted Weight")
            # Using st.success to display the result in a green box
            st.success(f"**The predicted weight of the fish is: {prediction[0]:.2f} grams**")
            st.balloons()
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("The model is not loaded. Cannot make predictions.")

st.sidebar.info(
    "**Note:** The accuracy of the prediction depends on the underlying model "
    "trained on the Fish Market dataset."
)

