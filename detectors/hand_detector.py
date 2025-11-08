"""
MediaPipe Hands를 사용한 손 인식 모듈

공식 문서: https://google.github.io/mediapipe/solutions/hands.html
"""

import cv2
import mediapipe as mp


class HandDetector:
    """
    MediaPipe Hands를 래핑한 손 인식 클래스
    
    손의 21개 랜드마크 포인트를 실시간으로 감지합니다:
    - 0: 손목 (WRIST)
    - 1-4: 엄지손가락 (THUMB)
    - 5-8: 검지손가락 (INDEX_FINGER)
    - 9-12: 중지손가락 (MIDDLE_FINGER)
    - 13-16: 약지손가락 (RING_FINGER)
    - 17-20: 새끼손가락 (PINKY)
    """
    
    def __init__(self, 
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        HandDetector 초기화
        
        Args:
            max_num_hands (int): 감지할 최대 손 개수 (기본값: 2)
            min_detection_confidence (float): 손 감지 최소 신뢰도 (0.0~1.0, 기본값: 0.5)
            min_tracking_confidence (float): 손 추적 최소 신뢰도 (0.0~1.0, 기본값: 0.5)
        
        Example:
            >>> detector = HandDetector(max_num_hands=1)
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # MediaPipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        print(f"✓ HandDetector 초기화 완료 (최대 손 개수: {max_num_hands})")
    
    def detect(self, image):
        """
        이미지에서 손을 감지합니다.
        
        Args:
            image (numpy.ndarray): BGR 포맷의 입력 이미지
        
        Returns:
            results: MediaPipe Hands 감지 결과
                - multi_hand_landmarks: 감지된 손들의 랜드마크 리스트
                - multi_handedness: 왼손/오른손 정보
        
        Example:
            >>> detector = HandDetector()
            >>> results = detector.detect(frame)
            >>> if results.multi_hand_landmarks:
            ...     print(f"감지된 손 개수: {len(results.multi_hand_landmarks)}")
        """
        # BGR을 RGB로 변환 (MediaPipe는 RGB를 사용)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 성능 향상을 위해 이미지를 쓰기 불가능으로 설정
        image_rgb.flags.writeable = False
        
        # 손 감지 수행
        results = self.hands.process(image_rgb)
        
        # 이미지를 다시 쓰기 가능으로 설정
        image_rgb.flags.writeable = True
        
        return results
    
    def draw_landmarks(self, image, results):
        """
        감지된 손 랜드마크를 이미지에 그립니다.
        
        Args:
            image (numpy.ndarray): 랜드마크를 그릴 이미지
            results: detect() 메서드의 반환 결과
        
        Returns:
            numpy.ndarray: 랜드마크가 그려진 이미지
        
        Example:
            >>> detector = HandDetector()
            >>> results = detector.detect(frame)
            >>> frame = detector.draw_landmarks(frame, results)
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손 랜드마크 그리기
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return image
    
    def get_hand_info(self, results, image_shape):
        """
        감지된 손의 정보를 추출합니다.
        
        Args:
            results: detect() 메서드의 반환 결과
            image_shape: 이미지의 shape (height, width, channels)
        
        Returns:
            list: 각 손의 정보를 담은 딕셔너리 리스트
                [{
                    'handedness': 'Left' or 'Right',
                    'landmarks': [(x1, y1), (x2, y2), ...],  # 픽셀 좌표
                    'confidence': 0.95
                }]
        
        Example:
            >>> detector = HandDetector()
            >>> results = detector.detect(frame)
            >>> hands_info = detector.get_hand_info(results, frame.shape)
            >>> for hand in hands_info:
            ...     print(f"{hand['handedness']} 손 감지 (신뢰도: {hand['confidence']:.2f})")
        """
        hands_info = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            height, width, _ = image_shape
            
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # 랜드마크를 픽셀 좌표로 변환
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    landmarks.append((x, y))
                
                hand_info = {
                    'handedness': handedness.classification[0].label,
                    'landmarks': landmarks,
                    'confidence': handedness.classification[0].score
                }
                hands_info.append(hand_info)
        
        return hands_info
    
    def close(self):
        """
        리소스를 해제합니다.
        
        Example:
            >>> detector = HandDetector()
            >>> # ... 사용 ...
            >>> detector.close()
        """
        self.hands.close()
        print("✓ HandDetector 리소스 해제 완료")

