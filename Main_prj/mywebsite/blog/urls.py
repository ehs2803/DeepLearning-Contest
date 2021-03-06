from django.urls import path
from blog import views


urlpatterns =[
    path("", views.main, name="main"),                          # main화면 url 연결
    path("about/", views.about, name="about"),                  # About화면 url 연결
    path("Drowsiness/", views.Drowsiness, name="Drowsiness"),   # 졸음감지 url 연결
    path("Blinking/", views.Blinking, name="Blinking"),         # 눈깜빡임 url 연결
    path('video_feed1', views.video_feed1, name='video_feed1'),
    path('video_feed2', views.video_feed2, name='video_feed2')
]