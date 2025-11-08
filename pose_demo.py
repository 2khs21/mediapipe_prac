"""
전신 포즈 감지 데모 스크립트

MediaPipe Pose를 사용하여 실시간으로 전신의 33개 랜드마크를 감지하고
자세를 화면에 표시합니다.

실행 방법:
    python pose_demo.py

종료 방법:
    ESC 키 또는 'q' 키
"""

import cv2
from detectors.pose_detector import PoseDetector


def main():
    """
    전신 포즈 감지 데모 메인 함수
    """
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("카메라 초기화 완료")
    
    # 포즈 detector 초기화
    detector = PoseDetector(
        static_image_mode=False,         # 비디오 모드
        model_complexity=1,              # 중간 복잡도 (균형)
        smooth_landmarks=True,           # 랜드마크 스무딩
        enable_segmentation=False,       # 분할 마스크 비활성화 (성능 향상)
        min_detection_confidence=0.5,    # 감지 신뢰도 50%
        min_tracking_confidence=0.5      # 추적 신뢰도 50%
    )
    
    window_name = 'Pose Detection Demo'
    
    print("\n" + "="*50)
    print("전신 포즈 감지 데모가 시작되었습니다!")
    print("="*50)
    print("전신이 보이도록 카메라에서 떨어지세요")
    print("종료: ESC 키 또는 'q' 키")
    print("")
    print("키 조작:")
    print("  1: 랜드마크 포인트 토글")
    print("  2: 연결선 토글")
    print("="*50 + "\n")
    
    # 그리기 옵션
    draw_landmarks = True
    draw_connections = True
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)
            
            # 포즈 감지
            results = detector.detect(frame)
            
            # 포즈 그리기
            if draw_landmarks or draw_connections:
                frame = detector.draw_landmarks(
                    frame, 
                    results,
                    draw_landmarks=draw_landmarks,
                    draw_connections=draw_connections
                )
            
            # 포즈 정보 추출
            pose_info = detector.get_pose_info(results, frame.shape)
            
            # 화면 상단에 정보 표시
            if pose_info:
                info_text = "Pose: DETECTED"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                landmarks_text = f"Landmarks: {pose_info['num_landmarks']}"
                cv2.putText(frame, landmarks_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # 평균 가시성 점수
                avg_visibility = sum(pose_info['visibility_scores']) / len(pose_info['visibility_scores'])
                visibility_text = f"Visibility: {avg_visibility:.2f}"
                cv2.putText(frame, visibility_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                info_text = "Pose: NOT DETECTED"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 그리기 옵션 상태 표시
            options_y = frame.shape[0] - 60
            option1 = f"1: Landmarks {'ON' if draw_landmarks else 'OFF'}"
            option2 = f"2: Connections {'ON' if draw_connections else 'OFF'}"
            
            cv2.putText(frame, option1, (10, options_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, option2, (10, options_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 화면에 표시
            cv2.imshow(window_name, frame)
            
            # 키 입력 확인
            key = cv2.waitKey(1) & 0xFF
            
            # ESC 키 또는 'q' 키로 종료
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("\n종료 키가 눌렸습니다.")
                break
            
            # 그리기 옵션 토글
            elif key == ord('1'):
                draw_landmarks = not draw_landmarks
                print(f"랜드마크: {'ON' if draw_landmarks else 'OFF'}")
            elif key == ord('2'):
                draw_connections = not draw_connections
                print(f"연결선: {'ON' if draw_connections else 'OFF'}")
    
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

