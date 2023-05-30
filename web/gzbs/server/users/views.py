from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework.views import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt import authentication
from rest_framework_simplejwt.views import TokenViewBase
from rest_framework import status
from rest_framework import permissions
from .serializers import MyTokenSerializer

# hoho_todo
def listShops(requests):
    return HttpResponse("this is shop list")

class DetailsView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [authentication.JWTAuthentication]

    def get(self, request, *args, **kwargs):
        print(f"authenticate: {request.successful_authenticator.authenticate(request)}")

        header = request.successful_authenticator.get_header(request)
        raw_token = request.successful_authenticator.get_raw_token(header)
        validated_token = request.successful_authenticator.get_validated_token(raw_token)
        print(f"token信息：{validated_token}")
        print(f"登录用户：{request.successful_authenticator.get_user(validated_token)}")

        return Response("get ok")
    
    def post(self, request, *args, **kwargs):
        return Response("post ok")
    

class LoginView(TokenViewBase):
    serializer_class = MyTokenSerializer

    def post(self, request, *args, **kwargs):
        # print(f'hoho: request: {request.data}')
        serializer = self.get_serializer(data = request.data)

        # print(f'hoho: serializer: {serializer}')


        try:
            serializer.is_valid(raise_exception = True)
        except Exception as e:
            raise ValueError(f"验证失败：{e}")
        
        return Response(serializer.validated_data, status = status.HTTP_200_OK)