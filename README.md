\# Dynamic Cyclone Early Warning System using AI



This project implements a lightweight cyclone movement prediction system using historical cyclone track data and deep learning.



The system predicts short-term cyclone trajectories using an LSTM model trained on IBTrACS cyclone records from the North Indian Ocean basin.



\## Features



• Cyclone trajectory prediction using LSTM  

• Uses only latitude and longitude data  

• Displacement-based forecasting  

• RMSE ≈ 0.93° (~104 km)  

• Interactive dashboard using Plotly Dash  



\## Dataset



IBTrACS – International Best Track Archive for Climate Stewardship  

https://www.ncei.noaa.gov/products/international-best-track-archive



\## Technologies



Python  

TensorFlow / Keras  

Pandas  

NumPy  

Scikit-learn  

Plotly Dash  



\## Project Workflow



Dataset → Data Cleaning → Sequence Creation → Model Training → Evaluation → Dashboard Visualization



\## Run the Dashboard



python dashboard/app\_v3.py



Open in browser:



http://127.0.0.1:8050

