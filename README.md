#  Delhi House Price Prediction (End-to-End ML Project)

An end-to-end Machine Learning project that predicts house prices in Delhi using property features and location-based clustering.

This project includes:
- Data preprocessing
- Feature engineering
- Location clustering (KMeans)
- Log transformation of target
- Random Forest regression model
- Streamlit web application

---

##  Project Objective

To build a complete machine learning pipeline that:

- Cleans and preprocesses real estate data
- Extracts meaningful features
- Uses geographical clustering for better price prediction
- Trains a regression model
- Deploys predictions through a Streamlit web app

---

##  Features Used for Prediction

The model predicts house prices based on:

- Area (sqft)
- Bedrooms
- Bathrooms
- Balcony
- Parking
- Property Status (Ready / Under Construction)
- Furnishing Status
- Type of Building
- Property Type (New / Resale)
- Location Cluster (Derived using KMeans)

---

##  Machine Learning Pipeline

1. Data Cleaning
2. Handling Missing Values
3. Feature Scaling (StandardScaler)
4. Location Clustering (KMeans with 4 clusters)
5. Log Transformation of Target Variable
6. Train-Test Split (80-20)
7. RandomForestRegressor Model
8. Pipeline Integration
9. Model Saving using Joblib

---

##  Location Clustering

Latitude and longitude values were scaled and grouped into 4 clusters using KMeans.

This helps the model understand area-wise price patterns instead of relying on raw coordinates.

---

##  Streamlit Web Application

The trained model is integrated into a Streamlit interface where users can:

- Enter property details
- Select categorical options
- Get real-time predicted house price

---

##  How To Run This Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/asmitsinghml/Delhi-House-Prediction.git
cd Delhi-House-Prediction
```

---

### Step 2: Install Required Libraries

```bash
pip install -r requirements.txt
```

If requirements.txt is not available, manually install:

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

---

### Step 3: Train the Model (Optional)

If model.pkl is not present:

```bash
python main_complete.py
```

This will train and save the model.

---

### Step 4: Run Streamlit App

```bash
streamlit run main_complete.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

##  Project Structure

```
Delhi-House-Prediction/
│
├── main_complete.py
├── end_on.ipynb
├── end_to_end.ipynb
├── Delhi_v2.csv
├── model.pkl
└── README.md
```

---

##  Model Performance

The model typically predicts within ₹10–12 lakh deviation depending on property configuration and cluster.

---

##  Future Improvements

- Add model comparison (XGBoost, LightGBM)
- Hyperparameter tuning
- Add interactive location map
- Deploy on Streamlit Cloud
- Add Docker support

---

##  Author

**Asmit Singh**  
B.Tech CSE Student  
Machine Learning & Data Science Enthusiast  

GitHub: https://github.com/asmitsinghml
