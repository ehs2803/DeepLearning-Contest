from django.urls import path
from blog import views


urlpatterns =[
    path("", views.main, name="main"),                          # main화면 url 연결
    path("about/", views.about, name="about"),                  # About화면 url 연결
    path("TaskManager/", views.Task_Manager, name="TaskManager"),
    path("Drowsiness/", views.Drowsiness, name="Drowsiness"),   # 졸음감지 url 연결
    path("Blinking/", views.Blinking, name="Blinking"),         # 눈깜빡임 url 연결
    path('task_manager', views.task_manager, name='task_manager'),
    path('sleep_detector', views.sleep_detector, name='sleep_detector'),
    path('blink_detector', views.blink_detector, name='blink_detector')
]