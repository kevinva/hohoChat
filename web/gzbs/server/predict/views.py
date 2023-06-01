from django.shortcuts import render
from rest_framework.views import APIView, Response
from rest_framework import permissions
from rest_framework_simplejwt import authentication

class CUBatchFatiguePredictionView(APIView):

    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [authentication.JWTAuthentication]

    def get(self, request, *args, **kwargs):
        print(f'[hoho: CUBatchFatiguePredictionView-get] request: {request.data}, args: {args}, kwargs: {kwargs}')

        return Response("get ok")

    def post(self, request, *args, **kwargs):
        print(f'[hoho: CUBatchFatiguePredictionView-post] request: {request.data}, args: {args}, kwargs: {kwargs}')

        return Response("get ok")
