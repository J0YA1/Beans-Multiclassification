import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("svc_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Dry Bean Classification")
st.write("Enter the bean measurements below.")

# Input features
Area = st.number_input("Area", min_value=0.00000)
Perimeter = st.number_input("Perimeter", min_value=0.00000)
MajorAxisLength = st.number_input("MajorAxisLength", min_value=0.0000)
MinorAxisLength = st.number_input("MinorAxisLength", min_value=0.0000)
AspectRation = st.number_input("AspectRation", min_value=0.0000, format= '%.5f')
Eccentricity = st.number_input("Eccentricity", min_value=0.0000, format= '%.5f')
ConvexArea = st.number_input("ConvexArea", min_value=0.0000)
EquivDiameter = st.number_input("EquivDiameter", min_value=0.0000)
Extent = st.number_input("Extent", min_value=0.0000, format ='%.5f')
Solidity = st.number_input("Solidity", min_value=0.0000, format = '%.5f')
Roundness = st.number_input("Roundness", min_value=0.0000, format = '%.5f')
Compactness = st.number_input("Compactness", min_value=0.0000, format = '%.5f')
ShapeFactor1 = st.number_input("ShapeFactor1", min_value=0.0000, format = '%.5f')
ShapeFactor2 = st.number_input("ShapeFactor2", min_value=0.0000, format = '%.5f')
ShapeFactor3 = st.number_input("ShapeFactor3", min_value=0.0000, format = '%.5f')
ShapeFactor4 = st.number_input("ShapeFactor4", min_value=0.0000, format = '%.5f')

if st.button("Predict Bean Class"):

    input_data = pd.DataFrame({
        'Area':[Area],
        'Perimeter':[Perimeter],
        'MajorAxisLength':[MajorAxisLength],
        'MinorAxisLength':[MinorAxisLength],
        'AspectRation':[AspectRation],
        'Eccentricity':[Eccentricity],
        'ConvexArea':[ConvexArea],
        'EquivDiameter':[EquivDiameter],
        'Extent':[Extent],
        'Solidity':[Solidity],
        'roundness':[Roundness],
        'Compactness':[Compactness],
        'ShapeFactor1':[ShapeFactor1],
        'ShapeFactor2':[ShapeFactor2],
        'ShapeFactor3':[ShapeFactor3],
        'ShapeFactor4':[ShapeFactor4]
    })

    prediction = model.predict(input_data)

    bean_class = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Bean Class: {bean_class[0]}")