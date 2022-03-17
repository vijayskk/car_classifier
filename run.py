import joblib

model = joblib.load('model.joblib')

pred = model.predict()