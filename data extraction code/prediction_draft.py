# for the final prediction model 
import tensorflow as tf
import joblib

model_h1 = tf.keras.models.load_model("../models/final_model_h1.keras")
scaler_h1 = joblib.load("../models/scaler_h1.pkl")
