from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import StandardScaler
from src.model import RegressionModel
from src.database import SessionLocal, Prediction
from datetime import datetime
import numpy as np
import torch
import pickle



# temporarily changes:
from src.database import Base, engine
Base.metadata.create_all(bind=engine)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load model once
try:
    model = RegressionModel(in_features=6)
    
    with open(r"checkpoints/model_epoch_45.pkl", "rb") as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    model.eval()  
    
    # Load scalers
    with open(r"checkpoints/scalerX.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(r"checkpoints/scalerY.pkl", "rb") as f:
        scaler_y = pickle.load(f)
        
except FileNotFoundError as e:
    print(f"Error loading required files: {e}. Make sure that model.pkl, scaler_X.pkl, and scaler_y.pkl are present.")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_result(request: Request,
                         age: float = Form(...),
                         bmi: float = Form(...),
                         sex: str = Form(...),
                         children: int = Form(...),
                         smoker: str = Form(...),
                         region: str = Form(...)):
    
    #Create Mappings
    sex_map = {"female" : 0, "male" : 1}
    smoker_map = {"no": 0, "yes": 1}
    region_map = {"southeast": 0, "southwest" : 1, "northeast" : 2, "northwest" : 3}
    
    
    #Create Encodings
    sex_encoded = sex_map.get(sex.lower(), 0)
    smoker_encoded = smoker_map.get(smoker.lower(), 0)
    region_encoded = region_map.get(region.lower(), 0)
    
    #Scaling
    X_to_scale = np.array([[age,bmi]], dtype=np.float32)
    X_scaled = scaler_X.transform(X_to_scale)
    
    #Get the age scaled and the bmi scaled
    age_scaled = X_scaled[:, 0].reshape(-1, 1) # First column (age)
    bmi_scaled = X_scaled[:, 1].reshape(-1, 1) # Second column (bmi)
    
    #Change the unscaled features into 2d arrays, duct tape solution but it works
    sex_arr = np.array([[sex_encoded]], dtype=np.float32)
    children_arr = np.array([[children]], dtype=np.float32)
    smoker_arr = np.array([[smoker_encoded]], dtype=np.float32)
    region_arr = np.array([[region_encoded]], dtype=np.float32)
    
    X_all = np.concatenate((age_scaled, sex_arr, bmi_scaled, children_arr, smoker_arr, region_arr), axis=1)
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    
    with torch.inference_mode():
        y_pred = model(X_tensor).numpy()
    y_pred_scaled = scaler_y.inverse_transform(y_pred.reshape(-1,1))
    pred_value = float(y_pred_scaled[0][0])
    #Save to PostgreSQL
    db = SessionLocal()
    new_record = Prediction(
        age=age,
        bmi=bmi,
        sex=sex,
        children=children,
        smoker=smoker,
        region=region,
        predicted_cost=pred_value,
        timestamp=datetime.now()
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    db.close()
    
    #Return
    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "prediction": round(y_pred_scaled[0][0], 2),
            "age": age,
            "bmi": bmi,
            "sex": sex,
            "smoker": smoker,
            "region": region
        }
    )


@app.get("/records")
async def get_records():
    db = SessionLocal()
    data = db.query(Prediction).all()
    db.close()
    return data
