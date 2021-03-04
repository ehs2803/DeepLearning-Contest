from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from blog.sleep import Sleep_Detector
from blog.sleep import Blink_Detector


# Create your views here.
def main(req):
    context = {

    }
    return render(req, "main.html", context=context)


def about(req):
    context = {

    }
    return render(req, "about.html", context=context)


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


def video_feed1(request):
    return StreamingHttpResponse(gen(Sleep_Detector()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def video_feed2(request):
    return StreamingHttpResponse(gen(Blink_Detector()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
