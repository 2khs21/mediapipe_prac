"""
얼굴 인식 데모 스크립트

MediaPipe Face Detection을 사용하여 실시간으로 얼굴을 감지하고
바운딩 박스와 주요 포인트를 화면에 표시합니다.

실행 방법:
    python face_demo.py

종료 방법:
    ESC 키 또는 'q' 키
"""

import cv2
from detectors.face_detector import FaceDetector


def main():
    """
    얼굴 인식 데모 메인 함수
    """
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("카메라 초기화 완료")
    
    # 얼굴 인식 detector 초기화
    detector = FaceDetector(
        min_detection_confidence=0.7,  # 감지 신뢰도 70%
        model_selection=0               # 단거리 모델 (2m 이내, 빠름)
    )
    
    window_name = 'Face Detection Demo'
    
    print("\n" + "="*50)
    print("얼굴 인식 데모가 시작되었습니다!")
    print("="*50)
    print("카메라를 보세요")
    print("종료: ESC 키 또는 'q' 키")
    print("="*50 + "\n")
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)
            
            # 얼굴 감지
            results = detector.detect(frame)
            
            # 감지 결과 그리기
            frame = detector.draw_detections(frame, results)
            
            # 얼굴 정보 추출
            faces_info = detector.get_face_info(results, frame.shape)
            
            # 화면 상단에 정보 표시
            info_text = f"Faces detected: {len(faces_info)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 각 얼굴의 신뢰도 표시
            for idx, face in enumerate(faces_info):
                confidence_text = f"Face {idx+1}: {face['confidence']:.2f}"
                y_position = 70 + idx * 40
                cv2.putText(frame, confidence_text, (10, y_position),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 화면에 표시
            cv2.imshow(window_name, frame)
            
            # 키 입력 확인
            key = cv2.waitKey(1) & 0xFF
            
            # ESC 키 또는 'q' 키로 종료
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("\n종료 키가 눌렸습니다.")
                break
    
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    
    finally:
        # 리소스 해제
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("모든 리소스가 해제되었습니다.")
        print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()

