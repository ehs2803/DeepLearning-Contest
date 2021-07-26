from django.http.response import StreamingHttpResponse
from TaskManager.sleep import Sleep_Detector
from TaskManager.sleep import Blink_Detector
from TaskManager.sleep import sleep_Blink_Detector
from TaskManager.models import *

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.hashers import check_password

from django.utils import timezone

# sleep.py 에서 사용자 ID 값 참조를 위한 전역변수
ID = None
USERNAME = None

# 회원 가입
def signup(request):
    global errorMsg     # 에러메시지
    # POST 요청 시 입력된 데이터(사용자 정보) 저장
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm = request.POST['confirm']
        firstname = request.POST['firstname']
        lastname = request.POST['lastname']

        # 회원가입
        try:
            # 회원가입 실패 시
            if not (username and password and confirm and firstname and lastname and email):
                errorMsg = '빈칸이 존재합니다!'
            elif password != confirm:
                errorMsg = '비밀번호가 일치하지 않습니다!'
            # 회원가입 성공 시
            else:
                User.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                    first_name=firstname,
                    last_name=lastname,
                    date_joined=timezone.now()
                ).save()
                return redirect('')         # 회원가입 성공했다는 메시지 출력 후 로그인 페이지로 이동
        except:
            errorMsg = '빈칸이 존재합니다!'
        return render(request, 'signup.html', {'error': errorMsg})
    # 회원가입 성공 후 이동
    return render(request, 'signup.html')


# 로그인
def login(request):
    global errorMsg         # 에러메시지
    # POST 요청시 입력된 데이터 저장
    if request.method == 'POST':                                        # 로그인 버튼 클릭
        username = request.POST['username']
        password = request.POST['password']
        try:
            if not (username and password):                             # 아이디/비밀번호 중 빈칸이 존재할 때
                errorMsg = '아이디/비밀번호를 입력하세요.'
            else:                                                       # 아이디/비밀번호 모두 입력됐을 때
                user = User.objects.get(username=username)              # 등록된 아이디의 정보 가져오기
                if check_password(password, user.password):             # 등록된 아이디의 비밀번호가 맞으면
                    request.session['id'] = user.id                     # 세션에 번호 추가
                    request.session['username'] = user.username         # 세션에 아이디 추가
                    request.session['email'] = user.email                   # 세션에 이메일 추가
                    request.session['first_name'] = user.first_name         # 세션에 이름 추가
                    request.session['last_name'] = user.last_name           # 세션에 성 추가
                    return redirect('main/')
                else:                                                   # 등록된 아이디의 비밀번호가 틀리면
                    errorMsg = '비밀번호가 틀렸습니다.'
        except:                                                         # 등록된 아이디의 정보가 없을 때
            errorMsg = '가입하지 않은 아이디 입니다.'

        return render(request, 'login.html', {'error': errorMsg})   # 에러 메세지와 로그인 페이지(login.html) 리턴
    # GET 요청시
    return render(request, 'login.html')                            # 로그인 페이지(login.html) 리턴


# 로그아웃
def logout(request):
    # 세션에 사용자 정보 존재할 경우
    if request.session.get('id'):
        del(request.session['id'])          # 사용자 번호 제거
        del(request.session['username'])    # 사용자 아이디 제거
    return redirect('/')            # 메인 페이지(index.html) 리턴


# 메인 페이지
def main(request):
    id = None
    username = None
    global ID, USERNAME
    if request.session.get('id', None):
        id = request.session.get('id', None)
        username = request.session.get('username', None)
        # DB 활용을 위한 전역변수 저장
        ID = id
        USERNAME = username
    # html로 세션 데이터 전송
    context = {
        'id' : id,            # 사용자 번호
        'username': username  # 사용자 아이디
    }
    return render(request, "main.html", context=context)


# About 페이지
def about(request):
    return render(request, "about.html")


# 마이페이지 임시
def MyPage(request):
    id = None
    username = None

    if request.session.get('id'):
        id = AuthUser.objects.get(id=request.session.get('id', None))
        username = AuthUser.objects.get(username=request.session.get('username', None))
    context = {
        'id':id,
        'username':username,
    }
    return render(request, 'mypage.html', context=context)


# 통합 페이지
def Task_Manager(request):
    return render(request, "TaskManager.html")


# 졸음 감지 페이지
def Drowsiness(request):
    return render(request, "Drowsiness.html")


# 눈 깜빡임 감지 페이지
def Blinking(request):
    return render(request, "Blinking.html")


# 게시판 페이지
def Board(request):
    id = None
    username = None
    if request.session.get('id'):
        id = request.session.get('id', None)
        username = request.session.get('username', None)
    context = {
        'id':id,
        'username':username
    }
    return render(request, "Board.html", context=context)


# 졸음 해소 스트레칭 동영상 페이지
def tip(request):
    return render(request, "tip.html")


# 카메라 연결
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
