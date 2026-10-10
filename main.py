import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 1. YOLOv8 최신/경량 모델 로드 (YOLO11n도 사용 가능)
model = YOLO("yolov8n.pt")

# 2. 리얼센스 파이프라인 및 설정 구성
pipeline = rs.pipeline()
config = rs.config()

# RGB와 Depth 스트림 활성화 (D435 추천 해상도 및 FPS)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
pipeline.start(config)

# 3. 중요: 뎁스 영상을 컬러 영상 좌표계에 강제로 '정렬(Align)'하는 객체 생성
# 이 과정을 거쳐야 RGB 이미지의 (x, y) 픽셀 위치가 Depth 이미지의 (x, y) 위치와 딱 맞아떨어집니다.
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # 프레임 세트 대기 및 가져오기
        frames = pipeline.wait_for_frames()
        
        # RGB와 Depth 프레임 맞춤 정렬
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # 이미지 데이터 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 4. YOLO 모델로 객체 인식 수행
        results = model(color_image, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 바운딩 박스 좌표 추출 (정수형 변환)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # 5. 객체의 중심점(Center Pixel) 계산
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 6. 중심점의 실제 거리(Distance) 추출 (미터 단위 반환)
                # get_distance(x, y) 함수가 내부 캘리브레이션 값을 이용해 정확한 거리를 계산해 줍니다.
                distance = depth_frame.get_distance(cx, cy)

                # 화면에 시각화 (바운딩 박스, 레이블, 거리 표시)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                
                # 거리 텍스트 포맷 (예: "person: 1.45m")
                text = f"{label}: {distance:.2f}m"
                cv2.putText(color_image, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 결과 화면 출력
        cv2.imshow("RealSense YOLO Object Detection", color_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 파이프라인 종료 및 창 닫기
    pipeline.stop()
    cv2.destroyAllWindows()