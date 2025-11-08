"""
얼굴 메시 감지 데모 스크립트

MediaPipe Face Mesh를 사용하여 실시간으로 얼굴의 468개 랜드마크를 감지하고
상세한 얼굴 메시를 화면에 표시합니다.

실행 방법:
    python face_mesh_demo.py

종료 방법:
    ESC 키 또는 'q' 키
"""

import cv2
from detectors.face_mesh_detector import FaceMeshDetector


def main():
    """
    얼굴 메시 감지 데모 메인 함수
    """
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("카메라 초기화 완료")
    
    # 얼굴 메시 detector 초기화
    detector = FaceMeshDetector(
        max_num_faces=1,                # 최대 1개의 얼굴 감지
        refine_landmarks=True,          # 눈과 입술 정밀화
        min_detection_confidence=0.5,   # 감지 신뢰도 50%
        min_tracking_confidence=0.5     # 추적 신뢰도 50%
    )
    
    window_name = 'Face Mesh Demo'
    
    print("\n" + "="*50)
    print("얼굴 메시 감지 데모가 시작되었습니다!")
    print("="*50)
    print("카메라를 보세요 (468개 랜드마크)")
    print("종료: ESC 키 또는 'q' 키")
    print("")
    print("키 조작:")
    print("  1: 테셀레이션 토글 (면)")
    print("  2: 윤곽선 토글")
    print("  3: 홍채 토글")
    print("="*50 + "\n")
    
    # 그리기 옵션
    draw_tesselation = True
    draw_contours = True
    draw_irises = True
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)
            
            # 얼굴 메시 감지
            results = detector.detect(frame)
            
            # 메시 그리기
            frame = detector.draw_landmarks(
                frame, 
                results,
                draw_tesselation=draw_tesselation,
                draw_contours=draw_contours,
                draw_irises=draw_irises
            )
            
            # 얼굴 정보 추출
            faces_info = detector.get_face_mesh_info(results, frame.shape)
            
            # 화면 상단에 정보 표시
            info_text = f"Faces: {len(faces_info)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 랜드마크 개수 표시
            if faces_info:
                landmarks_text = f"Landmarks: {faces_info[0]['num_landmarks']}"
                cv2.putText(frame, landmarks_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 그리기 옵션 상태 표시
            options_y = frame.shape[0] - 90
            option1 = f"1: Tesselation {'ON' if draw_tesselation else 'OFF'}"
            option2 = f"2: Contours {'ON' if draw_contours else 'OFF'}"
            option3 = f"3: Irises {'ON' if draw_irises else 'OFF'}"
            
            cv2.putText(frame, option1, (10, options_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, option2, (10, options_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, option3, (10, options_y + 50), 
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
                draw_tesselation = not draw_tesselation
                print(f"테셀레이션: {'ON' if draw_tesselation else 'OFF'}")
            elif key == ord('2'):
                draw_contours = not draw_contours
                print(f"윤곽선: {'ON' if draw_contours else 'OFF'}")
            elif key == ord('3'):
                draw_irises = not draw_irises
                print(f"홍채: {'ON' if draw_irises else 'OFF'}")
    
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

