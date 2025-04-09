from flask import Flask, render_template, Response
import cv2
import torch

# Flask 애플리케이션 생성
app = Flask(__name__)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 웹캠 스트리밍 함수
def generate_frames():
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLOv5 모델에 입력할 이미지 전처리
        results = model(frame)

        # 결과에서 바운딩 박스, 클래스 이름, 신뢰도 추출
        for *box, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 프레임 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# 라우트 설정
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)