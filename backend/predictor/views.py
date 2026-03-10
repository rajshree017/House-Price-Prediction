# This file handles the ML prediction logic
# When frontend sends house data, this returns predicted price

import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Train model once when server starts
housing = fetch_california_housing()
X = housing.data
y = housing.target * 100000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

@api_view(['POST'])
def predict_price(request):
    try:
        data = request.data

        # Get values from frontend form
        features = [
            float(data['MedInc']),
            float(data['HouseAge']),
            float(data['AveRooms']),
            float(data['AveBedrms']),
            float(data['Population']),
            float(data['AveOccup']),
            float(data['Latitude']),
            float(data['Longitude']),
        ]

        # Scale and predict
        features_scaled = scaler.transform([features])
        predicted_price = model.predict(features_scaled)[0]

        return Response({
            'predicted_price': round(predicted_price, 2),
            'status': 'success'
        })

    except Exception as e:
        return Response({'error': str(e), 'status': 'error'}, status=400)