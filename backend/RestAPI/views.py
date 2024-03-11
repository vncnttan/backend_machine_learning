from django.shortcuts import render
from django.conf import settings
import pandas as pd
import os
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from django.http import HttpResponse
import joblib


prediction_dict = {0: 'Insufficient_Weight', 
                   1: 'Normal_Weight', 
                   2: 'Obesity_Type_I', 
                   3: 'Obesity_Type_II', 
                   4: 'Obesity_Type_III', 
                   5: 'Overweight_Level_I', 
                   6: 'Overweight_Level_II'}


def index(request):
    return HttpResponse('Hello, World!')

class Predict(views.APIView):
    def post(self, request):
        model_path = os.path.join(settings.MODEL_ROOT, 'rfcModel.pkl')

        with open(model_path, 'rb') as file:
            model = joblib.load(file)
            encoder = joblib.load(os.path.join(settings.MODEL_ROOT, 'encoder.joblib'))

        for entry in request.data:
            try:
                df = pd.DataFrame(entry, index=[0])
                df = encoder.transform(df)
                result = list(model.predict(df))
                
            except Exception as err:
                print("Error: ", err)
                return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        return Response(prediction_dict[result[0]], status=status.HTTP_200_OK)