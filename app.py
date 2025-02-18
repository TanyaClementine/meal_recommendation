import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from google.generativeai import configure, GenerativeModel
from numpy import random
import matplotlib.pyplot as plt

# Configure Generative AI
configure(api_key="AIzaSyAFlHn5F45jt1yU8_oSabjRAGCZCQGXwPQ")
genai_model = GenerativeModel("gemini-pro")


# Load dataset
df = pd.read_csv("mealdata.csv")

# Function to get user input
def get_user_input():
    age = st.number_input("Enter your age:", min_value=1)
    weight = st.number_input("Enter your weight (kg):", min_value=1.0)
    height = st.number_input("Enter your height (m):", min_value=0.1)
    health_issues = st.multiselect("Select your health issues:", ["None", "Diabetes", "Hypertension", "Obesity", "High Cholesterol", "Kidney Issues"])    
    dietary_preference = st.selectbox("Are you vegetarian or non-vegetarian?", ["Veg", "Non-Veg"])
    activity_level = st.selectbox("Enter your activity level", ["Sedentary", "Moderate", "Active"])
    goal = st.selectbox("Enter your goal", ["Weight Loss", "Maintenance", "Gain"])
    
    bmi = calculate_bmi(weight, height)
    return {
        "Age": age,
        "Weight": weight,
        "Height": height,
        "BMI": bmi,
        "Health Issues": health_issues,
        "Dietary Preference": dietary_preference,
        "Activity Level": activity_level,
        "Goal": goal
    }

def generate_meal_description(meal_name, ingredients):
    """Generates a description for the meal using GenAI."""
    prompt = f"Provide a one line short,unique and personlised description for '{meal_name}' made with {ingredients}"
    response = genai_model.generate_content(prompt)
    return response.text if response else "Description not available."

# Function to calculate BMI and health status
def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        status = "Underweight"
    elif 18.5 <= bmi < 24.9:
        status = "Normal weight"
    elif 25 <= bmi < 29.9:
        status = "Overweight"
    else:
        status = "Obese"
    
    return f"{bmi:.2f} ({status})"

# Preprocess user data
def preprocess_user_data(df, dietary_preference, health_issues):
    df = df[df["Veg_NonVeg"] == dietary_preference]
    
    # Apply health-based filtering
    if "Diabetes" in health_issues:
        df = df[df["Carbohydrate"] < 15]  # Limit carbs for diabetes
    if "Kidney Issues" in health_issues:
        df = df[df["Protein"] < 5]  # Low protein for kidney issues
    if "Hypertension" in health_issues:
        df = df[df["Sodium"] < 300] if "Sodium" in df.columns else df  # Low sodium
    if "High Cholesterol" in health_issues:
        df = df[df["Fat"] < 10]  # Low fat meals
    if "Obesity" in health_issues:
        df = df[df["Calories"] < 200]  # Low-calorie meals
    
    if df.empty:
        st.error("No matching meals found for your preference.")
        return None, None
    
    nutrition_columns = ["Calories", "Protein", "Carbohydrate", "Fat", "Fibre"]
    scaler = RobustScaler()
    return df, scaler

# Recommend meals based on user profile
def recommend_meals_user(df, user_input):
    df, scaler = preprocess_user_data(df, user_input["Dietary Preference"], user_input["Health Issues"])
    if df is None:
        return None

    nutrition_columns = ["Calories", "Protein", "Carbohydrate", "Fat", "Fibre"]
    
    goal_modifier = 50 if user_input["Goal"] == "Weight Loss" else (70 if user_input["Goal"] == "Gain" else 60)
    activity_modifier = 0 if user_input["Activity Level"] == "Sedentary" else (10 if user_input["Activity Level"] == "Moderate" else 20)
    
    user_profile = np.array([
        goal_modifier + activity_modifier,
        0.8 * user_input["Weight"],
        3 * user_input["Weight"],
        0.4 * user_input["Weight"],
        25
    ]).reshape(1, -1)
    user_profile = scaler.fit_transform(user_profile)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(df[nutrition_columns], np.arange(len(df)))
    df["Score"] = rf.predict(df[nutrition_columns])
    return df.sample(frac=1).sort_values(by="Score", ascending=False)

# Pantry-based meal recommendation
def get_pantry_input():
    pantry = set(st.text_input("Enter ingredients you have (comma-separated):").lower().split(', '))
    return pantry

def preprocess_pantry_data(df, pantry):
    if "Main_ingredients" not in df.columns:
        st.error("Error: 'Main_ingredients' column missing in dataset.")
        return pd.DataFrame()
    
    def ingredient_match(ingredients):
        meal_ingredients = set(map(str.strip, ingredients.lower().split(',')))
        return not meal_ingredients.isdisjoint(pantry)
    
    return df[df['Main_ingredients'].apply(ingredient_match)]

def recommend_meals_pantry(df, pantry, num_recommendations=3):
    df_filtered = preprocess_pantry_data(df, pantry)
    
    if df_filtered.empty:
        st.error("No meals found based on your pantry ingredients.")
        return None
    
    recommendations = {}
    for meal_type in ["Breakfast", "Lunch", "Dinner", "Snacks"]:
        meals = df_filtered[df_filtered["Meal_Type"] == meal_type].head(num_recommendations)
        recommendations[meal_type] = meals
    
    return recommendations
def generate_pie_chart(row):
    fig, ax = plt.subplots(figsize=(2, 2))  # Reduce figure size
    nutrients = [row["Calories"], row["Protein"], row["Carbohydrate"], row["Fat"], row["Fibre"]]
    labels = ["Calories", "Protein", "Carbs", "Fat", "Fibre"]
    
    wedges, _, autotexts = ax.pie(
        nutrients, autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white'}, pctdistance=0.75, textprops={'fontsize': 8}
    )
    
    # Move labels to a legend
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, frameon=False)    
    ax.set_title(f"{row['Dish_Name']} - Nutrient Breakdown", fontsize=10)
    st.pyplot(fig)
# Sidebar for filters
def display_sidebar():
    st.sidebar.header("Filters")
    meal_type = st.sidebar.selectbox("Select Meal Type", ["Breakfast", "Lunch", "Dinner", "Snacks"])
    
    return  meal_type

# Display recommendations in tabular format with clickable video, like/dislike buttons
def display_recommendations(df, meal_type, num_recommendations=3):
    df_filtered = df[df["Meal_Type"] == meal_type].head(num_recommendations)
    
    if df_filtered.empty:
        st.warning(f"No {meal_type} meals found matching your criteria.")
        return
    
    st.write(f"### {meal_type} Recommendations")
    
    for index, row in df_filtered.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])

            with col1:
                # Define placeholder before using it in col2
                description_placeholder = st.empty()  

            if st.button(f"🍽 {row['Dish_Name']}", key=f"dish_{index}"):
                st.markdown(f"[Preparation Video]({row['Preparing Video']})", unsafe_allow_html=True)
                generate_pie_chart(row)

            with col2:
                if st.button(f"👍 Like", key=f"like_{index}"):
                    meal_description = generate_meal_description(row['Dish_Name'], row['Main_ingredients'])
                    description_placeholder.write(f"**Description:** {meal_description}")

            with col3:
                if st.button(f"👎 Dislike", key=f"dislike_{index}"):
                    st.warning(f"Oops... You disliked {row['Dish_Name']}. 😕")


# Main Streamlit App
def main():
    st.title("Meal Recommendation System")
    
    # Sidebar for user inputs and filters
    user_details = get_user_input()
    pantry_items = get_pantry_input()
    meal_type = display_sidebar()
    
    # User-based meal recommendations
    recommended_meals_user = recommend_meals_user(df, user_details)
    if recommended_meals_user is not None:
        st.header(f"Recommended {meal_type}s based on your profile")
        display_recommendations(recommended_meals_user, meal_type, num_recommendations=3)
    
    # Pantry-based meal recommendations
    pantry_recommendations = recommend_meals_pantry(df, pantry_items)
    if pantry_recommendations and meal_type in pantry_recommendations:
        st.header(f"Recommended {meal_type}s based on your pantry")
        display_recommendations(pantry_recommendations[meal_type], meal_type, num_recommendations=3)
    
    # Summary of user profile
    st.subheader("Your Profile Summary")
    st.write(f"**Age:** {user_details['Age']}, **Weight:** {user_details['Weight']} kg, **Height:** {user_details['Height']} m")
    st.write(f"**BMI:** {user_details['BMI']}")
    st.write(f"**Dietary Preference:** {user_details['Dietary Preference']}")
    st.write(f"**Goal:** {user_details['Goal']}")

# Run the app
if __name__ == "__main__":
    main()
