import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

st.title("Data Annotations")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
      image = Image.open(uploaded_file)
      files = {"file": uploaded_file.getvalue()}
      response = requests.post("http://localhost:8000/predict", files=files)
      
      if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)

            df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].astype(int)
            df[['x_center', 'y_center', 'width', 'height']] = df[['x_center', 'y_center', 'width', 'height']].round(3) 
            df['confidence'] = df['confidence'].apply(lambda x: f'{x*100:.2f}%')
            
            selected_idx = st.multiselect("Select the index of rows to delete", df.index)
            if st.button('Delete'):
                  df = df.drop(index=selected_idx)

            draw = ImageDraw.Draw(image)

            options = list(df.index) + ["View All"]
            highlighted_option = st.selectbox("Select the index to highlight or view all", options)

            if highlighted_option == "View All":
                  for index, row in df.iterrows():
                        draw.rectangle([row['x1'], row['y1'], row['x2'], row['y2']], outline="green", width=2)
                        draw.text((row['x1'], row['y1']), row['class name'], fill="green")
            else:
                  row = df.loc[highlighted_option]
                  draw.rectangle([row['x1'], row['y1'], row['x2'], row['y2']], outline="orange", width=5)
                  draw.text((row['x1'], row['y1']), row['class name'], fill="orange")

            st.image(image, caption='Predictions.', use_column_width=True)
            st.dataframe(df)
      else:
            st.write(f"Failed to get response from the API. Status code: {response.status_code}")
