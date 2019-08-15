import cv2
import logging

logger = logging.getLogger('__main__')


def capture_image(img_path):
    """

    :param img_path: path of the image to be captured
    :return: None
    """
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        height, width = frame.shape[:2]
        s_len = 300
        ver_bl = (width // 2 - s_len // 2, height // 2 - s_len // 2)
        ver_tr = (width // 2 + s_len // 2, height // 2 + s_len // 2)
        
        output = frame.copy()
        cv2.rectangle(output, ver_bl, ver_tr, (0, 0, 0), 2)
        cv2.putText(output, 'Align puzzle with grid, then press SPACE BAR to capture image',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Futoshiki Solver Capture', output)
        
        if not ret:
            break
            
        k = cv2.waitKey(1)
    
        if k % 256 == 27:
            # ESC pressed
            logger.info('Escape hit, closing window')
            break
            
        elif k % 256 == 32:
            # SPACE pressed
            cv2.imwrite(img_path, frame)
            logger.info(f'{img_path} written')
            break
    
    cam.release()


def display_preview(img_path):
    """
    
    """
    img = cv2.imread(img_path)
    cv2.putText(img, 'Press SPACE BAR to accept image, press escape to take another',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Futoshiki Solver Capture', img)
    
    k = cv2.waitKey()
    
    if k % 256 == 27:
        # ESC pressed, take another
        logger.info('Retake image')
        return False
        
    elif k % 256 == 32:
        # SPACE pressed, accept image
        logger.info('Image accepted')
        return True


def get_puzzle(img_path):
    cv2.namedWindow('Futoshiki Solver Capture')
    
    while True:
        capture_image(img_path)
        result = display_preview(img_path)
        
        if result:
            cv2.destroyAllWindows()
            break
