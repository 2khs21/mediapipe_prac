"""
손 인식 데모 스크립트

MediaPipe Hands를 사용하여 실시간으로 손을 감지하고 
21개 랜드마크 포인트를 화면에 표시합니다.

실행 방법:
    python hand_demo.py

종료 방법:
    ESC 키 또는 'q' 키
"""

import cv2
from detectors.hand_detector import HandDetector


def main():
    """
    손 인식 데모 메인 함수
    """
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("✓ 카메라 초기화 완료")
    
    # 손 인식 detector 초기화
    detector = HandDetector(
        max_num_hands=2,              # 최대 2개의 손 감지
        min_detection_confidence=0.7,  # 감지 신뢰도 70%
        min_tracking_confidence=0.5    # 추적 신뢰도 50%
    )
    
    window_name = 'Hand Detection Demo'
    
    print("\n" + "="*50)
    print("손 인식 데모가 시작되었습니다!")
    print("="*50)
    print("카메라 앞에서 손을 보여주세요")
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
            
            # 손 감지
            results = detector.detect(frame)
            
            # 랜드마크 그리기
            frame = detector.draw_landmarks(frame, results)
            
            # 손 정보 표시
            hands_info = detector.get_hand_info(results, frame.shape)
            
            # 화면 상단에 정보 표시
            info_text = f"Hands detected: {len(hands_info)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 각 손의 정보 표시
            for idx, hand in enumerate(hands_info):
                text = f"{hand['handedness']} ({hand['confidence']:.2f})"
                y_position = 70 + idx * 40
                cv2.putText(frame, text, (10, y_position),
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

