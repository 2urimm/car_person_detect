import sys
import time
import cv2
import numpy as np
# 기존 PyQt6 또는 잘못된 PySide6 임포트를 지우고 아래 코드로 교체합니다.
from PySide6.QtCore import QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt  # PySide6에서 Qt는 단독으로 임포트하거나 이 방식을 씁니다.
import pyrealsense2 as rs
from ultralytics import YOLO


class CameraThread(QThread):
    # pyqtSignal -> Signal로 변경
    change_pixmap_signal = Signal(np.ndarray)
    fps_signal = Signal(float)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # 1. YOLO 모델 로드
        model = YOLO("yolov8n.pt")

        # 2. 리얼센스 카메라 설정
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline.start(config)

        # RGB-Depth 정렬 객체
        align = rs.align(rs.stream.color)

        prev_time = 0

        while self._run_flag:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # FPS 계산
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 30.0
            prev_time = current_time
            self.fps_signal.emit(fps)

            # 이미지 변환
            color_image = np.asanyarray(color_frame.get_data())

            # 3. YOLO 객체 탐지
            results = model(color_image, stream=True, verbose=False)

            for r in results:
                boxes = r.boxes
                for box in boxes: # Tensor를 완전히 파이썬 일반 리스트로 풀어서 변환
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls)
                    label = model.names[cls]

                    # 객체 중심점 거리 측정
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    distance = depth_frame.get_distance(cx, cy)

                    # 영상 데이터에 그리기 (BGR 기준)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                    
                    text = f"{label}: {distance:.2f}m"
                    cv2.putText(color_image, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # GUI 창으로 영상 프레임 전송
            self.change_pixmap_signal.emit(color_image)

        pipeline.stop()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSense D435 객체 인식 & 거리 측정 시스템")
        self.resize(700, 550)

        # 메인 레이아웃 구성
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 상단 실시간 FPS 및 상태 정보 레이블
        self.info_label = QLabel("카메라 초기화 중... | FPS: 0.0")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;")
        self.layout.addWidget(self.info_label)

        # 카메라 피드를 출력할 메인 비디오 레이블
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        # 카메라 처리를 백그라운드 스레드로 실행 (GUI가 버벅이지 않게 분리)
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.fps_signal.connect(self.update_fps)
        self.thread.start()

    def update_image(self, cv_img):
        """백그라운드에서 처리된 OpenCV 이미지를 받아 PyQt 화면에 출력"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        
        # OpenCV(BGR)를 PyQt(RGB) 데이터로 변경
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # QLabel 크기에 맞춰 이미지 스케일 조정 후 그리기
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        p = pixmap.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(p)

    def update_fps(self, fps):
        """실시간 프레임(FPS) 상단 정보 갱신"""
        self.info_label.setText(f"상태: 정상 작동 중 | 실시간 프레임: {fps:.1f} FPS")

    def closeEvent(self, event):
        """GUI 창이 닫힐 때 카메라 스레드 안전하게 종료"""
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())