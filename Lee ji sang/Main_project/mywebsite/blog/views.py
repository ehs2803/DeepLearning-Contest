from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from blog.sleep import VideoCamera1, VideoCamera2


# Create your views here.
def index(req):
    context ={

    }
    return render(req, "index.html", context=context)

def single(req):
    context = {

    }
    return render(req, "single.html", context=context)

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
	return StreamingHttpResponse(gen(VideoCamera1()),
					content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed2(request):
	return StreamingHttpResponse(gen(VideoCamera2()),
					content_type='multipart/x-mixed-replace; boundary=frame')