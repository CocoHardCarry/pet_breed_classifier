import streamlit as st
from fastai.vision.all import *

def extract_breed(file_name):
    file_path = Path(file_name)
    parts = file_path.stem.split("_")
    breed = "_".join(parts[:-1])

    if parts[0].islower():
        return "DOG - " + breed
    else:
        return "CAT - " + breed
    print(extract_breed("Bengal_173.jpg"))

pet_breed_model = load_learner("pet_breed_model.pkl")

def predict(file_name):
    img = PILImage.create(file_name)
    prediction, idx, accuracy = pet_breed_model.predict(img)

    for i in img:
        results = pet_breed_model.predict(file_name)
        print(results)
        breed = results[0]
        breed_index = results[1]
        accuracy = results[2][breed_index]
        if accuracy > 0.9:
            img_title = f"{breed} - {accuracy * 100}% confident."
        else:
            img_title = f"I am not sure what this is, it might be {breed} - {accuracy * 100}% confident."

    img = PILImage.create(img)
    img.show(title=img_title)

st.text("Cat vs Dog Classifier")
st.text("Built by Jayden Hang")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    prediction = predict(uploaded_file)
    st.image(uploaded_file, caption=prediction, use_column_width=True)

