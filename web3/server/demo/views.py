from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def index(request):
    return HttpResponse(f'请求路径:{request.path}, 请求方法:{request.method}')