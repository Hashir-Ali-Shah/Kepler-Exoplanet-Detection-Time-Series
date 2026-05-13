import pandas as pd 
import numpy as np 
import tensorflow as tf 
from sklearn .metrics import accuracy_score ,classification_report 
from scipy .ndimage import gaussian_filter1d 
def preprocess_data (file_path ):
    """
    Replicates the preprocessing steps used during training:
    1. Gaussian Smoothing (sigma=3)
    2. Row-wise Scaling
    3. Reshaping for RNN (samples, timesteps, 1)
    """
    print (f"Loading data from {file_path}...")
    data =pd .read_csv (file_path )
    y =data ['LABEL']-1 
    X =data .drop ('LABEL',axis =1 )
    print ("Applying Gaussian smoothing...")
    X_smoothed =gaussian_filter1d (X ,sigma =3 ,axis =1 )
    print ("Applying row-wise scaling...")
    mean =X_smoothed .mean (axis =1 ).reshape (-1 ,1 )
    std =X_smoothed .std (axis =1 ).reshape (-1 ,1 )
    X_scaled =(X_smoothed -mean )/(std +1e-8 )
    X_reshaped =np .expand_dims (X_scaled ,axis =2 )
    return X_reshaped ,y 
def evaluate_keras_model (model_path ,X_test ,y_test ):
    print (f"\n--- Evaluating Model: {model_path} ---")
    try :
        model =tf .keras .models .load_model (model_path ,compile =False )
    except Exception as e :
        print (f"Error loading model {model_path}: {e}")
        return 
    y_pred_probs =model .predict (X_test ,verbose =0 ).ravel ()
    y_pred =(y_pred_probs >0.5 ).astype (int )
    accuracy =accuracy_score (y_test ,y_pred )
    report =classification_report (y_test ,y_pred ,target_names =['Normal Star (0)','Exoplanet (1)'],zero_division =0 )
    print (f"Accuracy: {accuracy:.4f}")
    print ("Classification Report:")
    print (report )
if __name__ =="__main__":
    test_file ="exoTest.csv"
    models =["lstm_exoplanet.keras","gru_exoplanet.keras"]
    try :
        X_test ,y_test =preprocess_data (test_file )
        for model_file in models :
            evaluate_keras_model (model_file ,X_test ,y_test )
    except Exception as e :
        print (f"An error occurred: {e}")