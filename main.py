from src.model import RegressionModel
from src.train import ModelTrainer
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    df = pd.read_csv(r"E:\Documents\Codes\Python\dockerized NN regressor\data\insurance.csv")
    
    #Standarizing
    scaler_X = StandardScaler()
    df[["age", "bmi"]] = scaler_X.fit_transform(df[["age", "bmi"]])
    scaler_y = StandardScaler()
    df["charges"] = scaler_y.fit_transform(df[["charges"]].values)

    #Encoding
    df["sex"] = df["sex"].map({"female": 0, "male": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    df['region'] = df['region'].map({"southeast": 0, "southwest" : 1, "northeast" : 2, "northwest" : 3})

    trainer = ModelTrainer(df.to_numpy(), test_size=0.2, batch_size=32)
    
    input_dim = trainer.X_train.shape[1]
    model = RegressionModel(input_dim)
    
    trainer.train(device=device, model=model, lr=1e-4, epochs=50, batch_size=32)
    
if __name__ == "__main__":
    main()