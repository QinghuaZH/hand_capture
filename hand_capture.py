import cv2
import mediapipe as mp

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    screen_width = 1920  # 设定全屏宽度（可根据实际屏幕调整）
    screen_height = 1080  # 设定全屏高度（可根据实际屏幕调整）
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)  # 镜像翻转，使其更符合人眼观看习惯
        frame = cv2.resize(frame, (screen_width, screen_height))  # 适应全屏显示
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in result.multi_hand_landmarks:
                hand_points = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_points.append((idx, cx, cy))
                    
                hand_data.append(hand_points)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print(hand_data)
        else:
            print("无双手")
        
        cv2.imshow("Hand Tracking", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("按下 q，程序退出")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
