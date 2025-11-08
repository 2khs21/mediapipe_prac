"""
MediaPipe Face Detection을 사용한 얼굴 인식 모듈

공식 문서: https://google.github.io/mediapipe/solutions/face_detection.html
"""

import cv2
import mediapipe as mp


class FaceDetector:
    """
    MediaPipe Face Detection을 래핑한 얼굴 인식 클래스
    
    얼굴을 감지하고 다음 정보를 제공합니다:
    - 얼굴 바운딩 박스
    - 6개의 주요 포인트 (눈, 코, 입, 귀)
    - 감지 신뢰도 점수
    """
    
    def __init__(self, 
                 min_detection_confidence=0.5,
                 model_selection=0):
        """
        FaceDetector 초기화
        
        Args:
            min_detection_confidence (float): 얼굴 감지 최소 신뢰도 (0.0~1.0, 기본값: 0.5)
            model_selection (int): 모델 선택
                - 0: 2미터 이내의 얼굴 감지 (빠름)
                - 1: 5미터 이내의 얼굴 감지 (정확함)
        
        Example:
            >>> detector = FaceDetector(min_detection_confidence=0.7)
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        
        # MediaPipe Face Detection 초기화
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
            model_selection=self.model_selection
        )
        
        model_name = "단거리(2m)" if model_selection == 0 else "장거리(5m)"
        print(f"✓ FaceDetector 초기화 완료 (모델: {model_name})")
    
    def detect(self, image):
        """
        이미지에서 얼굴을 감지합니다.
        
        Args:
            image (numpy.ndarray): BGR 포맷의 입력 이미지
        
        Returns:
            results: MediaPipe Face Detection 감지 결과
                - detections: 감지된 얼굴들의 정보 리스트
        
        Example:
            >>> detector = FaceDetector()
            >>> results = detector.detect(frame)
            >>> if results.detections:
            ...     print(f"감지된 얼굴 개수: {len(results.detections)}")
        """
        # BGR을 RGB로 변환 (MediaPipe는 RGB를 사용)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 성능 향상을 위해 이미지를 쓰기 불가능으로 설정
        image_rgb.flags.writeable = False
        
        # 얼굴 감지 수행
        results = self.face_detection.process(image_rgb)
        
        # 이미지를 다시 쓰기 가능으로 설정
        image_rgb.flags.writeable = True
        
        return results
    
    def draw_detections(self, image, results):
        """
        감지된 얼굴 정보를 이미지에 그립니다.
        
        Args:
            image (numpy.ndarray): 얼굴 정보를 그릴 이미지
            results: detect() 메서드의 반환 결과
        
        Returns:
            numpy.ndarray: 얼굴 정보가 그려진 이미지
        
        Example:
            >>> detector = FaceDetector()
            >>> results = detector.detect(frame)
            >>> frame = detector.draw_detections(frame, results)
        """
        if results.detections:
            for detection in results.detections:
                # 얼굴 바운딩 박스 및 주요 포인트 그리기
                self.mp_drawing.draw_detection(
                    image,
                    detection
                )
        
        return image
    
    def get_face_info(self, results, image_shape):
        """
        감지된 얼굴의 상세 정보를 추출합니다.
        
        Args:
            results: detect() 메서드의 반환 결과
            image_shape: 이미지의 shape (height, width, channels)
        
        Returns:
            list: 각 얼굴의 정보를 담은 딕셔너리 리스트
                [{
                    'bbox': (x, y, w, h),  # 바운딩 박스 (픽셀 좌표)
                    'confidence': 0.95,    # 감지 신뢰도
                    'keypoints': {         # 주요 포인트 (픽셀 좌표)
                        'right_eye': (x, y),
                        'left_eye': (x, y),
                        'nose_tip': (x, y),
                        'mouth_center': (x, y),
                        'right_ear_tragion': (x, y),
                        'left_ear_tragion': (x, y)
                    }
                }]
        
        Example:
            >>> detector = FaceDetector()
            >>> results = detector.detect(frame)
            >>> faces_info = detector.get_face_info(results, frame.shape)
            >>> for face in faces_info:
            ...     print(f"얼굴 신뢰도: {face['confidence']:.2f}")
        """
        faces_info = []
        
        if results.detections:
            height, width, _ = image_shape
            
            for detection in results.detections:
                # 바운딩 박스 정보
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # 주요 포인트 (keypoints)
                keypoints = {}
                if detection.location_data.relative_keypoints:
                    kp_names = [
                        'right_eye',
                        'left_eye', 
                        'nose_tip',
                        'mouth_center',
                        'right_ear_tragion',
                        'left_ear_tragion'
                    ]
                    
                    for idx, kp in enumerate(detection.location_data.relative_keypoints):
                        if idx < len(kp_names):
                            kp_x = int(kp.x * width)
                            kp_y = int(kp.y * height)
                            keypoints[kp_names[idx]] = (kp_x, kp_y)
                
                face_info = {
                    'bbox': (x, y, w, h),
                    'confidence': detection.score[0],
                    'keypoints': keypoints
                }
                faces_info.append(face_info)
        
        return faces_info
    
    def close(self):
        """
        리소스를 해제합니다.
        
        Example:
            >>> detector = FaceDetector()
            >>> # ... 사용 ...
            >>> detector.close()
        """
        self.face_detection.close()
        print("✓ FaceDetector 리소스 해제 완료")

