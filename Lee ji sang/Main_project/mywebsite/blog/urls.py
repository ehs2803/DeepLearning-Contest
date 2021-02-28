from django.urls import path
from blog import views

urlpatterns =[
    path("", views.index, name="index"), #main화면 url 연결
    path("single/", views.single, name="single"), #test url 연결
    path("Drowsiness/", views.Drowsiness, name="Drowsiness"), #졸음감지 url 연결
    path("Blinking/", views.Blinking, name="Blinking"), #눈깜빡임 url 연결
    path('video_feed', views.video_feed, name='video_feed')
]