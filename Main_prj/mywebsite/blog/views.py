from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from blog.sleep import Sleep_Detector
from blog.sleep import Blink_Detector
from blog.sleep import sleep_Blink_Detector


# Create your views here.
def main(req):
    context = {

    }
    return render(req, "main.html", context=context)


def about(req):
    context = {

    }
    return render(req, "about.html", context=context)


def Task_Manager(req):
    context = {

    }
    return render(req, "TaskManager.html", context=context)


def Drowsiness(req):
    context = {

    }
    return render(req, "Drowsiness.html", context=context)


def Blinking(req):
    context = {

    }
    return render(req, "Blinking.html", context=context)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def task_manager(request):
    return StreamingHttpResponse(gen(sleep_Blink_Detector()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def sleep_detector(request):
    return StreamingHttpResponse(gen(Sleep_Detector()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def blink_detector(request):
    return StreamingHttpResponse(gen(Blink_Detector()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
