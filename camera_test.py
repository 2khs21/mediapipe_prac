"""
맥북 카메라 동작 확인용 기본 스크립트
ESC 키를 누르면 종료됩니다.
"""

import cv2


def initialize_camera(camera_id=0):
    """
    카메라를 초기화합니다.
    
    Args:
        camera_id (int): 카메라 장치 ID (기본값: 0)
    
    Returns:
        cv2.VideoCapture: 초기화된 카메라 객체
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise Exception(f"카메라 {camera_id}를 열 수 없습니다.")
    
    print(f"카메라 {camera_id} 초기화 완료")
    return cap


def main():
    """
    카메라를 켜고 실시간 비디오를 표시합니다.
    """
    # 카메라 초기화
    cap = initialize_camera(0)
    
    print("카메라가 켜졌습니다. ESC 키를 눌러 종료하세요.")
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 화면에 표시
            cv2.imshow('Camera Test', frame)
            
            # ESC 키 입력 시 종료 (27은 ESC의 ASCII 코드)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        # 리소스 해제
        cap.release()
        cv2.destroyAllWindows()
        print("카메라를 종료했습니다.")


if __name__ == "__main__":
    main()

