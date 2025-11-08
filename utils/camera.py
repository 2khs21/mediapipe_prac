"""
카메라 관련 유틸리티 함수들
OpenCV를 사용한 카메라 초기화, 프레임 읽기, 리소스 해제 등의 기능 제공
"""

import cv2


def initialize_camera(camera_id=0):
    """
    카메라를 초기화합니다.
    
    Args:
        camera_id (int): 카메라 장치 ID (기본값: 0 - 기본 웹캠)
    
    Returns:
        cv2.VideoCapture: 초기화된 카메라 객체
    
    Raises:
        Exception: 카메라를 열 수 없을 때 발생
    
    Example:
        >>> cap = initialize_camera(0)
        >>> print(cap.isOpened())
        True
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise Exception(f"카메라 {camera_id}를 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
    
    print(f"✓ 카메라 {camera_id} 초기화 완료")
    return cap


def release_camera(cap):
    """
    카메라 리소스를 해제하고 모든 윈도우를 닫습니다.
    
    Args:
        cap (cv2.VideoCapture): 해제할 카메라 객체
    
    Example:
        >>> cap = initialize_camera(0)
        >>> release_camera(cap)
        ✓ 카메라 리소스 해제 완료
    """
    if cap is not None and cap.isOpened():
        cap.release()
    
    cv2.destroyAllWindows()
    print("✓ 카메라 리소스 해제 완료")


def read_frame(cap):
    """
    카메라에서 한 프레임(이미지)을 읽습니다.
    
    Args:
        cap (cv2.VideoCapture): 카메라 객체
    
    Returns:
        tuple: (성공 여부(bool), 프레임 이미지(numpy.ndarray))
    
    Example:
        >>> cap = initialize_camera(0)
        >>> ret, frame = read_frame(cap)
        >>> if ret:
        ...     print(f"프레임 크기: {frame.shape}")
    """
    ret, frame = cap.read()
    return ret, frame


def show_frame(window_name, frame):
    """
    프레임을 화면에 표시합니다.
    
    Args:
        window_name (str): 윈도우 이름
        frame (numpy.ndarray): 표시할 이미지
    
    Example:
        >>> cap = initialize_camera(0)
        >>> ret, frame = read_frame(cap)
        >>> if ret:
        ...     show_frame("My Camera", frame)
    """
    cv2.imshow(window_name, frame)


def wait_key(delay=1):
    """
    키 입력을 기다립니다.
    
    Args:
        delay (int): 대기 시간(밀리초). 0이면 무한 대기
    
    Returns:
        int: 눌린 키의 ASCII 코드 (하위 8비트)
    
    Example:
        >>> key = wait_key(1)
        >>> if key == 27:  # ESC
        ...     print("ESC 키가 눌렸습니다")
    """
    return cv2.waitKey(delay) & 0xFF

