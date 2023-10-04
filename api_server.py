from fastapi import FastAPI,UploadFile, File
import torch,io
from PIL import Image
import pandas as pd

app = FastAPI()

custom_model = torch.hub.load('ultralytics/yolov5', 'custom', path='lib/custom_model.pt', force_reload=True)
general_model = torch.hub.load('ultralytics/yolov5', 'custom', path='lib/general_model.pt', force_reload=True)
custom_model.eval()
general_model.eval()
without_classes = set(["traffic light", "stop sign", "parking meter", "bird", "scissors", "mouse", "tie", "clock", "suitcase", "umbrella", "cow", "airplane", "kite", "book", "bench"])
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    

    custom_results = custom_model(image)
    custom_xyxy = custom_results.pandas().xyxy[0]  
    custom_xywhn = custom_results.pandas().xywhn[0]  

    custom_data = pd.concat([custom_xywhn['name'], custom_xyxy[['xmin', 'ymin', 'xmax', 'ymax']], custom_xywhn[['xcenter', 'ycenter', 'width', 'height']], custom_xyxy['confidence']], axis=1)
    custom_data.columns = ['class name', 'x1', 'y1', 'x2', 'y2', 'x_center', 'y_center', 'width', 'height', 'confidence']

    general_results = general_model(image)
    general_xyxy = general_results.pandas().xyxy[0]  
    general_xywhn = general_results.pandas().xywhn[0]  

    filter_object = ~general_xywhn['name'].isin(without_classes)
    filtered_general_xyxy = general_xyxy[filter_object]
    filtered_general_xywhn = general_xywhn[filter_object]

    general_data = pd.concat([filtered_general_xywhn['name'], filtered_general_xyxy[['xmin', 'ymin', 'xmax', 'ymax']], filtered_general_xywhn[['xcenter', 'ycenter', 'width', 'height']], filtered_general_xyxy['confidence']], axis=1)
    general_data.columns = ['class name', 'x1', 'y1', 'x2', 'y2', 'x_center', 'y_center', 'width', 'height', 'confidence']
    # general_data = pd.concat([general_xywhn['name'], general_xyxy[['xmin', 'ymin', 'xmax', 'ymax']], general_xywhn[['xcenter', 'ycenter', 'width', 'height']], general_xyxy['confidence']], axis=1)
    # general_data.columns = ['class name', 'x1', 'y1', 'x2', 'y2', 'x_center', 'y_center', 'width', 'height', 'confidence']
    # general_df = general_data[~general_data['class name'].isin(without_classes)]
    # print(general_df)
    
    
    combined_data = pd.concat([custom_data, general_data], ignore_index=True)
    
    return combined_data.to_dict(orient="records")