import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="LifeGuard AI", page_icon="🩺", layout="wide")

# CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛡️ LifeGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Advanced Disease Prediction & Specialist Finder</p>", unsafe_allow_html=True)
st.divider()

# Symptoms
l1 = [
'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue',
'muscle_wasting','vomiting','burning_micturition','fatigue',
'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss',
'restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough',
'high_fever','sunken_eyes','breathlessness','sweating','dehydration',
'indigestion','headache','yellowish_skin','dark_urine','nausea',
'loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain',
'diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure',
'fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise',
'blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes',
'sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region',
'bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
'bruising','obesity','swollen_legs','puffy_face_and_eyes','enlarged_thyroid',
'brittle_nails','excessive_hunger','slurred_speech','knee_pain','hip_joint_pain',
'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness',
'spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
'loss_of_smell','bladder_discomfort',
'continuous_feel_of_urine','passage_of_gases','internal_itching',
'toxic_look_(typhos)','depression','irritability','muscle_pain',
'altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','watering_from_eyes',
'increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances',
'receiving_blood_transfusion','receiving_unsterile_injections','coma',
'stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
'blood_in_sputum','prominent_veins_on_calf','palpitations',
'painful_walking','pus_filled_pimples','blackheads','scurring',
'skin_peeling','silver_like_dusting','small_dents_in_nails',
'inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

# Medical Info
medical_info = {
"Fungal infection": {"type": "Skin Condition 🍄", "precautions": ["Keep skin dry", "Avoid tight clothes"], "doctor": "Dermatologist"},
    "Allergy": {"type": "Immune/Allergic Condition 🤧", "precautions": ["Avoid allergens", "Wear mask outdoors"], "doctor": "Allergist"},
    "GERD": {"type": "Digestive Condition 🥣", "precautions": ["Avoid spicy food", "Avoid lying down after meals"], "doctor": "Gastroenterologist"},
    "Chronic cholestasis": {"type": "Liver Condition 🍷", "precautions": ["Avoid alcohol", "Low fat diet"], "doctor": "Hepatologist"},
    "Drug Reaction": {"type": "Immune Response 💊", "precautions": ["Stop medication", "Consult doctor"], "doctor": "Dermatologist"},
    "Peptic ulcer disease": {"type": "Digestive Condition 🤢", "precautions": ["Avoid spicy food", "Regular meals"], "doctor": "Gastroenterologist"},
    "AIDS": {"type": "Immune System Condition 🎗️", "precautions": ["Safe practices", "Regular follow-up"], "doctor": "Infectious Disease Specialist"},
    "Diabetes": {"type": "Metabolic Condition 🩸", "precautions": ["Low sugar diet", "Daily exercise"], "doctor": "Endocrinologist"},
    "Gastroenteritis": {"type": "Digestive Condition 🤮", "precautions": ["Drink ORS", "Eat light food"], "doctor": "General Physician"},
    "Bronchial Asthma": {"type": "Respiratory Condition 🫁", "precautions": ["Avoid dust", "Keep inhaler handy"], "doctor": "Pulmonologist"},
    "Hypertension": {"type": "Circulatory Condition 🩺", "precautions": ["Low salt diet", "Reduce stress"], "doctor": "Cardiologist"},
    "Migraine": {"type": "Neurological Condition 🧠", "precautions": ["Dark room rest", "Avoid loud noise"], "doctor": "Neurologist"},
    "Cervical spondylosis": {"type": "Musculoskeletal Condition 🦴", "precautions": ["Neck exercises", "Proper pillow"], "doctor": "Orthopedic Surgeon"},
    "Paralysis (brain hemorrhage)": {"type": "Neurological Emergency ⚡", "precautions": ["Immediate hospitalization", "Physiotherapy"], "doctor": "Neurologist"},
    "Jaundice": {"type": "Hepatobiliary (Liver) Condition 🟡", "precautions": ["Rest", "Avoid oily food"], "doctor": "Hepatologist"},
    "Malaria": {"type": "Infectious Condition 🦟", "precautions": ["Mosquito net", "Stay hydrated"], "doctor": "Infectious Disease Specialist"},
    "Chicken pox": {"type": "Viral Infection 🌡️", "precautions": ["Isolation", "Oatmeal baths"], "doctor": "General Physician"},
    "Dengue": {"type": "Infectious Condition 🦟", "precautions": ["Check platelets", "Stay hydrated"], "doctor": "General Physician"},
    "Typhoid": {"type": "Bacterial Infection 💧", "precautions": ["Boiled water", "Light diet"], "doctor": "General Physician"},
    "Hepatitis A": {"type": "Liver Infection 🧼", "precautions": ["Avoid alcohol", "Clean food"], "doctor": "Hepatologist"},
    "Tuberculosis": {"type": "Respiratory Infection 😷", "precautions": ["Masking", "Finish medicine course"], "doctor": "Pulmonologist"},
    "Common Cold": {"type": "Respiratory Condition 🤧", "precautions": ["Steam inhalation", "Warm fluids"], "doctor": "General Physician"},
    "Pneumonia": {"type": "Respiratory Condition 🫁", "precautions": ["Chest physiotherapy", "Rest"], "doctor": "Pulmonologist"},
    "Heartattack": {"type": "Cardiovascular Emergency ❤️‍🔥", "precautions": ["Emergency call", "Chew aspirin"], "doctor": "Cardiologist"},
    "Hypothyroidism": {"type": "Metabolic Condition 🦋", "precautions": ["Regular checkup", "Healthy diet"], "doctor": "Endocrinologist"},
    "Hyperthyroidism": {"type": "Metabolic Condition 🦋", "precautions": ["Medication adherence", "Avoid stress"], "doctor": "Endocrinologist"},
    "Osteoarthritis": {"type": "Joint Condition 🦴", "precautions": ["Weight management", "Light exercise"], "doctor": "Orthopedic Surgeon"},
    "Arthritis": {"type": "Autoimmune/Joint Condition 👐", "precautions": ["Joint protection", "Warm compress"], "doctor": "Rheumatologist"},
    "Acne": {"type": "Skin Condition ✨", "precautions": ["Clean face", "Avoid oily cosmetics"], "doctor": "Dermatologist"},
    "Urinary tract infection": {"type": "Urinary Condition 🚽", "precautions": ["Drink water", "Hygiene"], "doctor": "Urologist"},
    "Psoriasis": {"type": "Chronic Skin Condition 🧤", "precautions": ["Moisturize", "Avoid triggers"], "doctor": "Dermatologist"}

}

# Data Loading and Training
@st.cache_resource
def train_models():
    try:
        df = pd.read_csv("Training.csv")
        le = LabelEncoder()
        df['prognosis'] = le.fit_transform(df['prognosis'])
        X = df[l1]
        y = df['prognosis']
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100).fit(X, y),
        }
        return models, le
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

models, le = train_models()

# UI Tabs
tab1, tab2, tab3 = st.tabs(["🩺 Consultance", "🏥 Nearby Hospitals", "🚨 Emergency"])

with tab1:
    st.header("Predict Disease based on Symptoms")
    if models is not None:
        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            model_option = st.selectbox("Select AI Model:", list(models.keys()))
        with col_sel2:
            user_symptoms = st.multiselect("Select your symptoms:", sorted(l1))

        if st.button("Consultance"):
            if user_symptoms:
                input_vector = np.zeros(len(l1))
                for s in user_symptoms:
                    if s in l1:
                        input_vector[l1.index(s)] = 1
                #confidence score
                probabilities = models[model_option].predict_proba([input_vector])[0]
                prediction_idx = np.argmax(probabilities)
                confidence_score = probabilities[prediction_idx] * 100
                predicted_disease = le.inverse_transform([prediction_idx])[0]


                st.success(f"### Predicted Condition: **{predicted_disease}**")

                st.subheader("⚠️ Risk Assessment & Urgency")
                
                
                critical_diseases = ["Heart attack", "Paralysis (brain hemorrhage)", "Pneumonia", "Dengue", "AIDS", "Tuberculosis", "Bronchial Asthma", "Hypertension" ]
                
                is_emergency_symptoms = "chest_pain" in user_symptoms and "sweating" in user_symptoms
                
                if predicted_disease in critical_diseases or is_emergency_symptoms:
                    urgency = "🔴 CRITICAL: IMMEDIATE ATTENTION REQUIRED"
                    color = "#FF0000" 
                    instruction = "This is a potentially life-threatening condition. Please visit an Emergency Room (ER) or call 102/108 immediately."
                elif confidence_score > 75:
                    urgency = "🟡 Moderate Urgency"
                    color = "#FFA500" 
                    instruction = "Consult a specialist within 24 hours."
                else:
                    urgency = "🟢 Low Urgency"
                    color = "#28a745"
                    instruction = "Follow precautions and consult a doctor if symptoms persist."

                # Visual Warning Box
                st.markdown(f"""
                    <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; background-color: {color}10;">
                        <h3 style="color: {color}; margin: 0;">{urgency}</h3>
                        <p style="color: #333; margin-top: 10px;"><b>Action:</b> {instruction}</p>
                        <p style="font-size: 12px; color: #666;">AI Confidence: {confidence_score:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)

                

                st.divider()
                prediction_idx = models[model_option].predict([input_vector])[0]
                predicted_disease = le.inverse_transform([prediction_idx])[0]
                
                st.success(f"### Predicted Condition: **{predicted_disease}**")
                info = medical_info.get(predicted_disease, {
                    "type": "General Health Condition 🏥", 
                    "precautions": ["Consult a specialist"], 
                    "doctor": "General Physician"
                })
                
                condition_type = info.get("type", "General Health Condition 🏥")

                # 3. Reasoning & Analysis Section
                st.markdown(f"#### 🔍 Reasoning & Analysis")
                
                # Cleaning symptoms text for display
                symptoms_str = ", ".join(user_symptoms).replace('_', ' ')
                
                # Logic based reasoning message
                reasoning_msg = f"Based on the presence of **{symptoms_str}**, the AI has identified this as a **{condition_type}**."
                
                st.info(reasoning_msg)
                
                st.divider()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📋 Recommended Precautions")
                    for p in info["precautions"]:
                        st.write(f"🔹 {p}")
                
                with c2:
                    st.subheader("👨‍⚕️ Specialist to Consult")
                    doc_type = info["doctor"]
                    st.info(f"Condition suggests visiting a: **{doc_type}**")
                    
                    # Correct Maps link
                    maps_url = f"https://www.google.com/maps/search/{doc_type.replace(' ', '+')}+near+me"
                    st.markdown(f'''
                        <a href="{maps_url}" target="_blank">
                            <button style="background-color: #28a745; color: white; padding: 12px; border: none; border-radius: 8px; cursor: pointer; width: 100%; font-size: 16px;">
                                📍 Find {doc_type} Near Me
                            </button>
                        </a>
                    ''', unsafe_allow_html=True)
                    #telemedicine button
                    google_search_url = f"https://www.google.com/search?q={doc_type.replace(' ', '+')}+near+me"
                    
                    st.markdown(f'''
                        <a href="{google_search_url}" target="_blank">
                            <button style="background-color: #4285F4; color: white; padding: 15px; border: none; border-radius: 8px; cursor: pointer; width: 100%; font-size: 18px; font-weight: bold; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); margin-bottom: 10px;">
                                 Find {doc_type}
                            </button>
                        </a>
                        <p style='text-align: center; font-size: 12px; color: #666;'>
                            See top-rated {doc_type}s with reviews & photos
                        </p>
                    ''', unsafe_allow_html=True)
                    
                    
                    maps_url = f"https://www.google.com/maps/search/{doc_type.replace(' ', '+')}+near+me"
                    st.markdown(f'''
                        <a href="{maps_url}" target="_blank" style="text-decoration: none;">
                            <div style="text-align: center; color: #34a853; font-weight: 500; font-size: 14px; margin-top: 5px;">
                                📍 Open in Google Maps instead
                            </div>
                        </a>
                    ''', unsafe_allow_html=True)
            else:
                st.error("Please select at least one symptom.")

with tab2:
    st.header("Search Nearby Medical Facilities")
    facility_query = st.text_input("Search for:", "Best Hospitals")
    facility_url = f"https://www.google.com/maps/search/{facility_query.replace(' ', '+')}+near+me"
    st.markdown(f'<a href="{facility_url}" target="_blank"><button style="background-color:#17a2b8; color:white; padding:15px; width:100%; border:none; border-radius:10px;">🔍 Find "{facility_query}" Near Me</button></a>', unsafe_allow_html=True)

with tab3:
    st.header("Emergency Helplines")
    col1, col2 = st.columns(2)
    with col1:
        st.error("🚑 Ambulance: 102")
        st.markdown("[📞 Click to Call 102](tel:102)")
    with col2:
        st.error("🚑 Emergency: 108")
        st.markdown("[📞 Click to Call 108](tel:108)")

st.divider()
st.caption("Disclaimer: This is not a medical diagnosis.")