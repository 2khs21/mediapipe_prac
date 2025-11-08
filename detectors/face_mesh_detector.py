"""
MediaPipe Face Mesh를 사용한 얼굴 메시 감지 모듈

공식 문서: https://google.github.io/mediapipe/solutions/face_mesh.html

Face Mesh는 468개의 3D 랜드마크를 감지하여 얼굴의 상세한 구조를 파악합니다.
"""

import cv2
import mediapipe as mp


class FaceMeshDetector:
    """
    MediaPipe Face Mesh를 래핑한 얼굴 메시 감지 클래스
    
    468개의 3D 랜드마크 포인트로 얼굴 전체를 상세하게 감지합니다:
    - 얼굴 윤곽 (Face Oval)
    - 눈썹 (Eyebrows)
    - 눈 (Eyes) - 홍채 포함
    - 코 (Nose)
    - 입술 (Lips) - 상세 윤곽
    - 얼굴 전체 메시
    """
    
    def __init__(self, 
                 max_num_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        FaceMeshDetector 초기화
        
        Args:
            max_num_faces (int): 감지할 최대 얼굴 개수 (기본값: 1)
            refine_landmarks (bool): 눈과 입술 주변 랜드마크 정밀화 (기본값: True)
            min_detection_confidence (float): 얼굴 감지 최소 신뢰도 (0.0~1.0, 기본값: 0.5)
            min_tracking_confidence (float): 얼굴 추적 최소 신뢰도 (0.0~1.0, 기본값: 0.5)
        
        Example:
            >>> detector = FaceMeshDetector(max_num_faces=2, refine_landmarks=True)
        """
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        refine_status = "활성화" if refine_landmarks else "비활성화"
        print(f"✓ FaceMeshDetector 초기화 완료")
        print(f"  - 최대 얼굴 수: {max_num_faces}")
        print(f"  - 랜드마크 정밀화: {refine_status}")
        print(f"  - 총 랜드마크: 468개 (refine_landmarks=True 시 478개)")
    
    def detect(self, image):
        """
        이미지에서 얼굴 메시를 감지합니다.
        
        Args:
            image (numpy.ndarray): BGR 포맷의 입력 이미지
        
        Returns:
            results: MediaPipe Face Mesh 감지 결과
                - multi_face_landmarks: 감지된 얼굴들의 랜드마크 리스트
        
        Example:
            >>> detector = FaceMeshDetector()
            >>> results = detector.detect(frame)
            >>> if results.multi_face_landmarks:
            ...     print(f"감지된 얼굴 개수: {len(results.multi_face_landmarks)}")
        """
        # BGR을 RGB로 변환 (MediaPipe는 RGB를 사용)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 성능 향상을 위해 이미지를 쓰기 불가능으로 설정
        image_rgb.flags.writeable = False
        
        # 얼굴 메시 감지 수행
        results = self.face_mesh.process(image_rgb)
        
        # 이미지를 다시 쓰기 가능으로 설정
        image_rgb.flags.writeable = True
        
        return results
    
    def draw_landmarks(self, image, results, draw_tesselation=True, draw_contours=True, draw_irises=True):
        """
        감지된 얼굴 메시를 이미지에 그립니다.
        
        Args:
            image (numpy.ndarray): 메시를 그릴 이미지
            results: detect() 메서드의 반환 결과
            draw_tesselation (bool): 얼굴 메시 테셀레이션 그리기 (기본값: True)
            draw_contours (bool): 얼굴 윤곽선 그리기 (기본값: True)
            draw_irises (bool): 홍채 그리기 (기본값: True)
        
        Returns:
            numpy.ndarray: 메시가 그려진 이미지
        
        Example:
            >>> detector = FaceMeshDetector()
            >>> results = detector.detect(frame)
            >>> frame = detector.draw_landmarks(frame, results)
        """
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 얼굴 메시 테셀레이션 (면)
                if draw_tesselation:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )
                
                # 얼굴 윤곽선
                if draw_contours:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                
                # 홍채
                if draw_irises:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                    )
        
        return image
    
    def get_face_mesh_info(self, results, image_shape):
        """
        감지된 얼굴 메시의 상세 정보를 추출합니다.
        
        Args:
            results: detect() 메서드의 반환 결과
            image_shape: 이미지의 shape (height, width, channels)
        
        Returns:
            list: 각 얼굴의 정보를 담은 딕셔너리 리스트
                [{
                    'landmarks': [(x, y, z), ...],  # 468개 또는 478개 포인트
                    'num_landmarks': 468
                }]
        
        Example:
            >>> detector = FaceMeshDetector()
            >>> results = detector.detect(frame)
            >>> faces_info = detector.get_face_mesh_info(results, frame.shape)
            >>> for face in faces_info:
            ...     print(f"랜드마크 개수: {face['num_landmarks']}")
        """
        faces_info = []
        
        if results.multi_face_landmarks:
            height, width, _ = image_shape
            
            for face_landmarks in results.multi_face_landmarks:
                # 모든 랜드마크를 픽셀 좌표로 변환
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    z = landmark.z  # 상대적 깊이 값
                    landmarks.append((x, y, z))
                
                face_info = {
                    'landmarks': landmarks,
                    'num_landmarks': len(landmarks)
                }
                faces_info.append(face_info)
        
        return faces_info
    
    def get_specific_landmarks(self, results, image_shape):
        """
        얼굴의 주요 랜드마크 포인트를 추출합니다.
        
        주요 포인트:
        - 오른쪽 눈: 33, 133, 160, 159, 158, 157, 173
        - 왼쪽 눈: 362, 263, 387, 386, 385, 384, 398
        - 입술 외곽: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
        - 얼굴 윤곽: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        
        Args:
            results: detect() 메서드의 반환 결과
            image_shape: 이미지의 shape (height, width, channels)
        
        Returns:
            list: 각 얼굴의 주요 랜드마크를 담은 딕셔너리 리스트
        """
        faces_info = []
        
        if results.multi_face_landmarks:
            height, width, _ = image_shape
            
            for face_landmarks in results.multi_face_landmarks:
                landmarks_list = face_landmarks.landmark
                
                def get_point(idx):
                    lm = landmarks_list[idx]
                    return (int(lm.x * width), int(lm.y * height))
                
                face_info = {
                    'right_eye_center': get_point(33),
                    'left_eye_center': get_point(263),
                    'nose_tip': get_point(1),
                    'mouth_center': get_point(13),
                    'chin': get_point(152)
                }
                faces_info.append(face_info)
        
        return faces_info
    
    def close(self):
        """
        리소스를 해제합니다.
        
        Example:
            >>> detector = FaceMeshDetector()
            >>> # ... 사용 ...
            >>> detector.close()
        """
        self.face_mesh.close()
        print("✓ FaceMeshDetector 리소스 해제 완료")

