import streamlit as st
import pandas as pd
import textwrap
from sklearn.model_selection import train_test_split

import openai
import guidance
import os

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import t2ebm
# auto-reload
#%load_ext autoreload
#%autoreload 2
openai.api_key = "YOUR_KEY"

llm = guidance.llms.OpenAI("gpt-3.5-turbo-16k") 

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

dataset_description = """Hi there, here is a general description of the data set on which I trained the model. This description is from kaggle:

Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

The task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, you're given a set of personal records recovered from the ship's damaged computer system.

Feature Descriptions:

PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
Destination - The planet the passenger will be debarking to.
Age - The age of the passenger.
VIP - Whether the passenger has paid for special VIP service during the voyage.
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
Name - The first and last names of the passenger.
"""

y_axis_description = "The y-axis depicts contributions in log-odds towards the outcome, that is the probability that the passenger was transported to another dimension."

# Streamlit UI components
st.title("EBM Model Training App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display uploaded data
    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Select dependent variable
    dependent_var = st.selectbox("Select Dependent Variable", df.columns)

    # Select independent variables
    independent_vars = st.multiselect("Select Independent Variables", df.columns)

    if st.button("Train EBM Model"):
        if dependent_var and independent_vars:
            # Prepare data
            X = df[independent_vars]
            y = df[dependent_var]

            # Train EBM model
            ebm = ExplainableBoostingClassifier()
            ebm.fit(X, y)

            # Display model summary
            st.subheader("EBM Model Summary")
            #st.write(ebm.score(X, y))

            model_description = t2ebm.llm_describe_ebm(llm, 
                                           ebm,
                                           dataset_description=dataset_description, 
                                           y_axis_description=y_axis_description) # by default this creates a 30 sentence summary
            model_desc = (textwrap.fill(model_description, 80))
            st.success("EBM Model Trained Successfully!")
            #st.write(model_desc)

    index_list = [f"{row['PassengerId']} - {row['Name']}" for _, row in df.iterrows()]

            # Dropdown to select an index value
    selected_index = st.selectbox("Select DataFrame Index", index_list)
    df['Index'] = df['PassengerId']+' - '+df['Name']

    if selected_index:
        st.success(f"Selected Index: {selected_index}")
        #filtered_df = df[df['Index'].str.contains(selected_index)]
        filtered_df = df.loc[df['Index'] == selected_index]
        st.dataframe(filtered_df)
        observation = filtered_df.iloc[0]
        # Convert the observation into a formatted text string
        observation_text = ", ".join([f"{col} {val}" for col, val in observation.iteritems()])
        #st.write(observation_text)
        question_text = """Describe in upto 75 why the passenger in the details in quotes '""" + observation_text+ """ had the transported value as mentioned before"""
        #st.write(model_desc)
        data_text = """The data is described in the quotes '""" + dataset_description + model_desc + """'"""
        prompt = question_text + data_text
        reply = get_completion(prompt, model="gpt-3.5-turbo")
        st.write(reply)


