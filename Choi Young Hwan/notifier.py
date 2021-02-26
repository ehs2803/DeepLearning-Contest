from gtts import gTTS
from playsound import playsound
import time

# 졸음 감지 함수(임시)
def Drowse_Detector(pred_l, pred_r):
    detection = False
    # 모델의 예측값이 0.2 이하를 졸음의 근거로함
    if pred_l < 0.2 and pred_r < 0.2:
        det_start = time.time()
        # 3초 이상 눈을 감고 있으면 확실한 졸음으로 감지
        if det_start > 3:
            detection = True
            return detection


# 졸음감지, 눈동자 깜빡임 경고 알림 함수
# opt 는 졸음감지, 눈동자 깜빡임의 여부 (0은 졸음, 1은 눈동자 깜빡임)
# detection 은 감지 여부 (True, False)
def notifier(pred_r, pred_l):
    if Drowse_Detector(pred_r, pred_l):  # 졸음감지인 경우
        # 감지 시 알림(우선은 mp3 파일 재생으로 대체함)
        playsound(tts_s_path)
    elif Eye_Blink_Counter(pred_r, pred_l):  # 눈동자 깜빡임 부족인 경우
        # 감지 시 알림(우선은 mp3 파일 재생으로 대체함)
        playsound(tts_b_path)


# 졸음 감지 텍스트
sleep_notification = '졸음이 감지되었습니다.'

# 눈동자 깜빡임 경고 텍스트
blink_notification = '현재 1분에 n번 깜빡이고 있어요. 눈 깜빡임 횟수가 부족합니다!' \
        '1분에 10~15회 이상 깜빡이세요!' \

# 졸음감지, 눈동자 깜빡임 측정 파일 경로
tts_s_path = 'notification1.mp3'
tts_b_path = 'notification2.mp3'

# 알림 음성 메세지 저장
tts_s = gTTS(text=sleep_notification, lang='ko')
tts_s.save(tts_s_path)

tts_b = gTTS(text=blink_notification, lang='ko')
tts_b.save(tts_b_path)

# 알림 음성 메세지 실행
print("졸음감지 메세지")
# sd.Beep(2000, 2000)
playsound(tts_s_path)
# sd.Beep(2000, 2000)
print("눈동자 깜빡임 경고 메세지")
playsound(tts_b_path, False)
