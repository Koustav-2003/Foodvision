from fastapi import FastAPI,UploadFile,File,HTTPException
from PIL import Image
import torch
import io

from backend.model import load_model
from backend.classes import CLASS_NAMES

app=FastAPI(title='FoodVision API',
            description='Food Image Classification using ConvNeXt Small',
            version='1.0.0')

model,transform=load_model()

@app.get('/')
def home():
    return{
        'message':'Welcome to FoodVision API'
    }
    
@app.get('/health')
def health_check():
    return {
        'status':'healthy',
        'device':str(next(model.parameters()).device)
    }
    
@app.post('/predict')
async def predict(file:UploadFile=File(...)):
    
    if file.content_type not in ['image/jpeg','image/png','image/webp']:
        raise HTTPException(
            status_code=400,
            detail='Please upload a JPEG,PNG or WebP image'
        )
        
    try:
        image_bytes=await file.read()
        image=Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor=transform(image).unsqueeze(0)
        device=next(model.parameters()).device
        image_tensor=image_tensor.to(device)
        
        with torch.inference_mode():
            outputs=model(image_tensor)
            probabilities=torch.softmax(outputs,dim=1)
            top_5_probabilities,top_5_indices=torch.topk(
                probabilities,k=5,dim=1
            )
            top_5_predictions=[]
            for probability,index in zip(
                top_5_probabilities[0],top_5_indices[0]
            ):
                top_5_predictions.append({
                    'class':CLASS_NAMES[index.item()],
                    'confidence':round(probability.item()*100,2)
                })
                
            return{
                    'predictions':top_5_predictions
                }
                
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f'Could not process Image:{str(e)}'
        )