import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load data
def load_data():
    df = pd.read_csv("C:\\Users\\Sai Sreya\\Downloads\\Migraine Prediction\\migraine_data.csv")
    return df

# Preprocess data
def preprocess_data(df):
    df = df[['Age', 'Duration', 'Frequency', 'Location', 'Character', 'Intensity', 'Nausea', 'Vomit',
             'Phonophobia', 'Photophobia', 'Visual', 'Sensory', 'Dysphasia', 'Dysarthria', 'Vertigo',
             'Tinnitus', 'Hypoacusis', 'Diplopia', 'Defect', 'Ataxia', 'Conscience', 'Paresthesia', 'DPF', 'Type']]
    df.dropna(inplace=True)
    df_2=df.drop(columns=["Type", "Age", "Duration", "Frequency", "Visual", "Sensory"])
    labels= [df_2[col].unique().tolist() for col in df_2.columns]
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df, labels

# Train model
def train_model(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_tuned=tune_params(model, X_train, y_train)
    model_tuned.fit(X_train,y_train)
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)  # Predict using the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    
    # Print accuracy in terminal
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model, scaler

# Model Tuning
def  tune_params(model, X, y):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    model_tuned = RandomForestClassifier(**best_params)
    return  model_tuned

# Save model
def save_model(model):
    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Load model
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


# Make prediction
def make_prediction(model, scaler, features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    migraine_types_info = {
        0: ("Not likely to suffer from migraines", ""),
        1: ("Migraine without aura", "Preventive measures:\n- Maintain regular sleep patterns\n- Stay hydrated\n- Identify and avoid trigger foods\n- Manage stress"),
        2: ("Sporadic hemiplegic migraine", "Preventive measures:\n- Avoid known triggers such as stress, bright lights, and certain foods\n- Consider preventive medications under medical supervision\n- Maintain a healthy lifestyle with regular exercise and adequate sleep"),
        3: ("Familial hemiplegic migraine", "Preventive measures:\n- Identify and avoid triggers\n- Monitor symptoms and consult healthcare professionals for appropriate treatment and preventive strategies"),
        4: ("Typical aura without migraine", "Preventive measures:\n- Maintain a healthy lifestyle with regular exercise and balanced nutrition\n- Manage stress and anxiety levels\n- Avoid known triggers such as bright lights and certain foods"),
        5: ("Other", "Preventive measures vary based on individual symptoms and triggers. It's essential to consult healthcare professionals for personalized recommendations."),
        6: ("Basilar-type aura", "Preventive measures:\n- Avoid known triggers such as stress, lack of sleep, and certain foods\n- Consider preventive medications under medical supervision\n- Maintain a headache diary to track symptoms and triggers"),
        # Add more descriptions and preventive measures for other migraine types if needed
    }
    migraine_info, prevention_info = migraine_types_info.get(prediction[0], ("Unknown", ""))
    return migraine_info, prevention_info

# Main function
def main():
    st.title('Migraine Type Prediction')
    st.markdown("""
    Migraine is a common neurological disorder characterized by recurrent episodes of headache, often accompanied by other symptoms such as nausea, vomiting, and sensitivity to light and sound. These episodes, known as migraine attacks, can vary in duration and severity, lasting from a few hours to several days and ranging from mild discomfort to debilitating pain
    """)
    # Additional information about migraines
    additional_info = """
    ### Additional Information

    #### Migraine Triggers:
    - Certain foods and beverages, such as aged cheese, chocolate, and alcohol, can trigger migraines in some individuals.
    - Hormonal changes, such as those occurring during menstruation or menopause, can trigger migraines in women.
    - Stress, anxiety, and other emotional factors are common triggers for migraines.
    - Environmental factors, such as bright lights, strong odors, or changes in weather, can trigger migraines in susceptible individuals.

    #### Migraine Types:
    - Migraine without aura: Characterized by headache without preceding aura symptoms.
    - Migraine with aura: Includes specific neurological symptoms (aura) before or during the headache phase.
    - Chronic migraine: Involves frequent migraine attacks (15 or more days per month) over a prolonged period.
    - Other types, such as hemiplegic migraine, basilar-type migraine, and vestibular migraine, have distinct features and symptoms.

    #### Treatment Options:
    - Pain-relieving medications (analgesics) such as acetaminophen, NSAIDs, or triptans can help alleviate migraine symptoms.
    - Preventive medications taken regularly can reduce the frequency and severity of migraine attacks.
    - Lifestyle changes, such as maintaining a regular sleep schedule, managing stress, and identifying triggers, can help manage migraines.
    - Alternative therapies like acupuncture, biofeedback, or relaxation techniques may also provide relief for some individuals.
    """

    # Display additional information in the sidebar
    st.sidebar.markdown(additional_info)

    # Load data
    df = load_data()
    
    # Preprocess data
    df_encoded, menu = preprocess_data(df)
    
    # Train model
    X = df_encoded.drop(columns=['Type'])
    y = df_encoded['Type']
    model, scaler = train_model(X, y)
    
    # Save model
    save_model(model)
    
    # User input
    st.subheader('Enter Data')
    object_df= df.select_dtypes(include=["object"])
    int_df = df.select_dtypes(include=["int64"])
    object_df.drop(columns="Type", inplace=True)
    feature_labels=object_df.columns.tolist()
    int_labels = int_df.columns.tolist()

    features = []
    #Integer Inputs
    for i in range (len(int_labels)):
        feature_value = st.text_input(f"Enter value for {int_labels[i]}: ", value='0')
        if feature_value:
            int_input=int(feature_value)
            features.append(int_input)

    #Object Inputs
    for label in range (len(feature_labels)):
        feature_value = st.selectbox(feature_labels[label], menu[label])
        feature_value = int(feature_value) if feature_value.isdigit() else 0
        features.append(feature_value)
    
    # Make prediction
    if st.button('Predict', key='predict_button'):
        loaded_model = load_model()
        migraine_info, prevention_info = make_prediction(loaded_model, scaler, features)
        st.subheader('**Prediction**')
        st.success(f"**{migraine_info}**")
        st.write(f"{prevention_info}")


if __name__ == '__main__':
    main()