"""
MediaPipe Pose를 사용한 전신 포즈 감지 모듈

공식 문서: https://google.github.io/mediapipe/solutions/pose.html

Pose는 33개의 3D 랜드마크를 감지하여 전신 자세를 파악합니다.
"""

import cv2
import mediapipe as mp


class PoseDetector:
    """
    MediaPipe Pose를 래핑한 전신 포즈 감지 클래스
    
    33개의 3D 랜드마크 포인트로 전신을 감지합니다:
    
    랜드마크 인덱스:
    - 0-10: 얼굴 (코, 눈, 귀 등)
    - 11-16: 상체 (어깨, 팔꿈치, 손목)
    - 17-22: 손 (양손의 엄지, 검지, 새끼손가락)
    - 23-28: 하체 (엉덩이, 무릎, 발목)
    - 29-32: 발 (발뒤꿈치, 발끝)
    """
    
    # 주요 랜드마크 인덱스 정의
    LANDMARK_NAMES = {
        0: 'NOSE',
        1: 'LEFT_EYE_INNER', 2: 'LEFT_EYE', 3: 'LEFT_EYE_OUTER',
        4: 'RIGHT_EYE_INNER', 5: 'RIGHT_EYE', 6: 'RIGHT_EYE_OUTER',
        7: 'LEFT_EAR', 8: 'RIGHT_EAR',
        9: 'MOUTH_LEFT', 10: 'MOUTH_RIGHT',
        11: 'LEFT_SHOULDER', 12: 'RIGHT_SHOULDER',
        13: 'LEFT_ELBOW', 14: 'RIGHT_ELBOW',
        15: 'LEFT_WRIST', 16: 'RIGHT_WRIST',
        17: 'LEFT_PINKY', 18: 'RIGHT_PINKY',
        19: 'LEFT_INDEX', 20: 'RIGHT_INDEX',
        21: 'LEFT_THUMB', 22: 'RIGHT_THUMB',
        23: 'LEFT_HIP', 24: 'RIGHT_HIP',
        25: 'LEFT_KNEE', 26: 'RIGHT_KNEE',
        27: 'LEFT_ANKLE', 28: 'RIGHT_ANKLE',
        29: 'LEFT_HEEL', 30: 'RIGHT_HEEL',
        31: 'LEFT_FOOT_INDEX', 32: 'RIGHT_FOOT_INDEX'
    }
    
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        PoseDetector 초기화
        
        Args:
            static_image_mode (bool): 정적 이미지 모드 (기본값: False - 비디오 모드)
            model_complexity (int): 모델 복잡도 0, 1, 2 (기본값: 1)
                - 0: 가벼움 (빠름)
                - 1: 중간 (균형)
                - 2: 무거움 (정확함)
            smooth_landmarks (bool): 랜드마크 스무딩 (기본값: True)
            enable_segmentation (bool): 사람 분할 마스크 활성화 (기본값: False)
            smooth_segmentation (bool): 분할 마스크 스무딩 (기본값: True)
            min_detection_confidence (float): 최소 감지 신뢰도 (0.0~1.0, 기본값: 0.5)
            min_tracking_confidence (float): 최소 추적 신뢰도 (0.0~1.0, 기본값: 0.5)
        
        Example:
            >>> detector = PoseDetector(model_complexity=1)
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        print(f"✓ PoseDetector 초기화 완료")
        print(f"  - 모델 복잡도: {model_complexity} (0=빠름, 1=균형, 2=정확)")
        print(f"  - 분할 마스크: {'활성화' if enable_segmentation else '비활성화'}")
        print(f"  - 총 랜드마크: 33개")
    
    def detect(self, image):
        """
        이미지에서 포즈를 감지합니다.
        
        Args:
            image (numpy.ndarray): BGR 포맷의 입력 이미지
        
        Returns:
            results: MediaPipe Pose 감지 결과
                - pose_landmarks: 감지된 포즈의 랜드마크 (33개)
                - pose_world_landmarks: 실제 3D 좌표
                - segmentation_mask: 사람 분할 마스크 (enable_segmentation=True일 때)
        
        Example:
            >>> detector = PoseDetector()
            >>> results = detector.detect(frame)
            >>> if results.pose_landmarks:
            ...     print("포즈 감지됨!")
        """
        # BGR을 RGB로 변환 (MediaPipe는 RGB를 사용)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 성능 향상을 위해 이미지를 쓰기 불가능으로 설정
        image_rgb.flags.writeable = False
        
        # 포즈 감지 수행
        results = self.pose.process(image_rgb)
        
        # 이미지를 다시 쓰기 가능으로 설정
        image_rgb.flags.writeable = True
        
        return results
    
    def draw_landmarks(self, image, results, draw_landmarks=True, draw_connections=True):
        """
        감지된 포즈를 이미지에 그립니다.
        
        Args:
            image (numpy.ndarray): 포즈를 그릴 이미지
            results: detect() 메서드의 반환 결과
            draw_landmarks (bool): 랜드마크 포인트 그리기 (기본값: True)
            draw_connections (bool): 연결선 그리기 (기본값: True)
        
        Returns:
            numpy.ndarray: 포즈가 그려진 이미지
        
        Example:
            >>> detector = PoseDetector()
            >>> results = detector.detect(frame)
            >>> frame = detector.draw_landmarks(frame, results)
        """
        if results.pose_landmarks:
            # 연결선 설정
            connections = self.mp_pose.POSE_CONNECTIONS if draw_connections else None
            
            # 랜드마크 스타일 설정
            landmark_style = self.mp_drawing_styles.get_default_pose_landmarks_style() if draw_landmarks else None
            
            # 그리기 (둘 중 하나라도 True면 그림)
            if draw_landmarks or draw_connections:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    connections,
                    landmark_drawing_spec=landmark_style
                )
        
        return image
    
    def draw_segmentation(self, image, results, alpha=0.5):
        """
        사람 분할 마스크를 이미지에 오버레이합니다.
        (enable_segmentation=True일 때만 사용 가능)
        
        Args:
            image (numpy.ndarray): 마스크를 그릴 이미지
            results: detect() 메서드의 반환 결과
            alpha (float): 마스크 투명도 (0.0~1.0, 기본값: 0.5)
        
        Returns:
            numpy.ndarray: 마스크가 오버레이된 이미지
        """
        if results.segmentation_mask is not None:
            import numpy as np
            
            # 마스크를 3채널로 변환
            mask = results.segmentation_mask
            mask_3channel = cv2.cvtColor((mask * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
            
            # 초록색 오버레이
            green_overlay = mask_3channel.copy()
            green_overlay[:, :, 0] = 0  # B
            green_overlay[:, :, 1] = 255  # G
            green_overlay[:, :, 2] = 0  # R
            
            # 알파 블렌딩
            image = cv2.addWeighted(image, 1, green_overlay, alpha, 0)
        
        return image
    
    def get_pose_info(self, results, image_shape):
        """
        감지된 포즈의 상세 정보를 추출합니다.
        
        Args:
            results: detect() 메서드의 반환 결과
            image_shape: 이미지의 shape (height, width, channels)
        
        Returns:
            dict or None: 포즈 정보를 담은 딕셔너리
                {
                    'landmarks': [(x, y, z, visibility), ...],  # 33개 포인트
                    'visibility_scores': [0.0~1.0, ...],  # 각 포인트의 가시성
                    'presence_score': 0.95  # 전체적인 존재 신뢰도
                }
        
        Example:
            >>> detector = PoseDetector()
            >>> results = detector.detect(frame)
            >>> pose_info = detector.get_pose_info(results, frame.shape)
            >>> if pose_info:
            ...     print(f"어깨 위치: {pose_info['landmarks'][11]}")
        """
        if not results.pose_landmarks:
            return None
        
        height, width, _ = image_shape
        landmarks = []
        visibility_scores = []
        
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # 상대적 깊이
            visibility = landmark.visibility  # 가시성 점수 (0~1)
            
            landmarks.append((x, y, z, visibility))
            visibility_scores.append(visibility)
        
        pose_info = {
            'landmarks': landmarks,
            'visibility_scores': visibility_scores,
            'num_landmarks': len(landmarks)
        }
        
        return pose_info
    
    def get_specific_landmarks(self, results, image_shape):
        """
        주요 신체 부위의 랜드마크를 추출합니다.
        
        Args:
            results: detect() 메서드의 반환 결과
            image_shape: 이미지의 shape (height, width, channels)
        
        Returns:
            dict or None: 주요 부위의 좌표
                {
                    'nose': (x, y),
                    'left_shoulder': (x, y),
                    'right_shoulder': (x, y),
                    'left_elbow': (x, y),
                    'right_elbow': (x, y),
                    ...
                }
        """
        if not results.pose_landmarks:
            return None
        
        height, width, _ = image_shape
        landmarks_list = results.pose_landmarks.landmark
        
        def get_point(idx):
            lm = landmarks_list[idx]
            return (int(lm.x * width), int(lm.y * height))
        
        specific = {
            'nose': get_point(0),
            'left_shoulder': get_point(11),
            'right_shoulder': get_point(12),
            'left_elbow': get_point(13),
            'right_elbow': get_point(14),
            'left_wrist': get_point(15),
            'right_wrist': get_point(16),
            'left_hip': get_point(23),
            'right_hip': get_point(24),
            'left_knee': get_point(25),
            'right_knee': get_point(26),
            'left_ankle': get_point(27),
            'right_ankle': get_point(28)
        }
        
        return specific
    
    def close(self):
        """
        리소스를 해제합니다.
        
        Example:
            >>> detector = PoseDetector()
            >>> # ... 사용 ...
            >>> detector.close()
        """
        self.pose.close()
        print("✓ PoseDetector 리소스 해제 완료")

