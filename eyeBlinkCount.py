import time
def eyeBlinkCount(pred_r, pred_l):
    global eye_count_min, start
    if pred_r < 0.1 and pred_l < 0.1:
        eye_count_min += 1
        time.sleep(0.1)
    if time.time() - start > 10:
        if eye_count_min < 5:
            start = time.time()
            eye_count_min = 0
            return True
        else:
            start = time.time()
            eye_count_min = 0
            return False
    else:
        return False