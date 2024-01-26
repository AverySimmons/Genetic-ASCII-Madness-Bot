import cv2
import pyautogui
import numpy as np
import math

WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT = 46, 112, 1200, 975
INNER_X, INNER_Y, INNER_WIDTH, INNER_HEIGHT = 90, 80, WINDOW_WIDTH-90, WINDOW_HEIGHT-80

def main():
    prev_frame = None
    player_pos = np.array([0, 0])
    prev_cursor_pos = []

    while True:
        screenshot = pyautogui.screenshot(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

        prev_frame, cursor_pos, player_points, code_pos = getObjects(prev_frame, frame)
        if len(player_points) > 0:
            player_pos = updatePlayerPos(player_points)

        real_cursor_pos, cursor_vel = calcCursorVel(prev_cursor_pos, cursor_pos)

        prev_cursor_pos = cursor_pos

        displayWindow(real_cursor_pos, cursor_vel, player_pos, code_pos, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def getNetworkInput(cur_num, prev_frame, prev_cursor_pos, prev_player_pos):
    inp = []

    screenshot = pyautogui.screenshot(region=(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT))
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    prev_frame, cursor_pos, player_points, code_pos = getObjects(prev_frame, frame)
    player_pos = prev_player_pos
    if len(player_points) > 0:
        player_pos = updatePlayerPos(player_points)

    real_cursor_pos, cursor_vel = calcCursorVel(prev_cursor_pos, cursor_pos)

    real_cursor_pos = getClosestOrder(real_cursor_pos, player_pos)

    cur = 0
    while cur < cur_num:
        if cur < len(real_cursor_pos):
            p_to_c = real_cursor_pos[cur] - player_pos
            c_to_p = player_pos - real_cursor_pos[cur]
            cur_dist = np.linalg.norm(p_to_c) / 1000
            cur_angle = math.atan2(real_cursor_pos[cur][0] - player_pos[0], real_cursor_pos[cur][1] - player_pos[1]) / math.pi
            vel_angle = angleBetweenVectors(cursor_vel[cur], c_to_p) / math.pi

            inp.extend([cur_dist, cur_angle, vel_angle])
        
        else:
            inp.extend([-2, -2, -1])
        
        cur += 1
    
    close_code = np.array([100,100])
    if len(code_pos) == 0:
        inp.append(0)
    else:
        code_pos = getClosestOrder(code_pos, player_pos)
        code_angle = math.atan2(code_pos[0][0] - player_pos[0], code_pos[0][1] - player_pos[1]) / math.pi
        inp.append(code_angle)
        close_code = code_pos[0]
    
    mid_x_dist = (player_pos[0] - 605)/501
    mid_y_dist = (player_pos[1] - 511)/333
    inp.extend([mid_x_dist, mid_y_dist])
    
    return inp, close_code, cursor_pos, player_pos, prev_frame

def angleBetweenVectors(A, B):
    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    if magnitude_A == 0 or magnitude_B == 0: return 0

    cosine_theta = dot_product / (magnitude_A * magnitude_B)
    angle_radians = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

    return angle_radians

def calcCursorVel(prev_cursor_pos, cursor_pos):
    cursor_vel = []
    real_cur_num = min(len(cursor_pos), len(prev_cursor_pos))
    mid = np.array([WINDOW_X+WINDOW_WIDTH/2, WINDOW_Y+WINDOW_HEIGHT/2])
    cursor_order = getClosestOrder(cursor_pos, mid)[:real_cur_num]
    for cur in cursor_order:
        closest = getClosestOrder(prev_cursor_pos, cur)
        cursor_vel.append(normalize(cur - closest[0]))
    return cursor_order, cursor_vel

def getClosestOrder(poses, point):
    def sortKey(e):
        return np.linalg.norm(point - e)
    return sorted(poses, key=sortKey)

def updatePlayerPos(player_points):
    new_pos = np.array([0, 0])
    for pos in player_points:
        new_pos += pos
    new_pos[0] /= len(player_points)
    new_pos[1] /= len(player_points)
    return new_pos

def displayWindow(cursor_pos, cursor_vel, player_pos, code_pos, frame):
    for i, pos in enumerate(cursor_pos):
        cv2.rectangle(frame, (pos[0]-10, pos[1]-10), (pos[0]+20, pos[1]+20), (255, 0, 0), 2)
        cv2.rectangle(frame, (pos[0], pos[1]), (pos[0] + int(cursor_vel[i][0] * 20), pos[1] + int(cursor_vel[i][1] * 20)), (255, 0, 255), 2)
        
    cv2.rectangle(frame, (player_pos[0]-10, player_pos[1]-10), (player_pos[0]+20, player_pos[1]+20), (0, 255, 0), 2)
        
    for pos in code_pos:
        cv2.rectangle(frame, (pos[0]-10, pos[1]-10), (pos[0]+20, pos[1]+20), (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Real-Time Object Detection", frame)

def getObjects(prev_frame, frame):

    # Convert the frame to grayscale for motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize prev_frame if not done yet
    if prev_frame is None:
        prev_frame = gray_frame

    # Convert the frame to HSV color space for better color filtering
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([255, 111, 255])

    # Define lower and upper bounds for red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the frame to get masks for white and red colors
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Find contours in the combined mask
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cursor_pos = []
    player_points = []
    code_pos = []

    for cnt in white_contours:
        cnt_pos = getCntPosition(cnt)
        if len(cnt_pos) == 0: continue
        if cv2.contourArea(cnt) > 100:
            cursor_pos.append(cnt_pos)
        elif cntIsMoving(cnt, gray_frame, prev_frame):
            if cnt_pos[0] > INNER_X and cnt_pos[0] < INNER_WIDTH and \
                cnt_pos[1] > INNER_Y and cnt_pos[1] < INNER_HEIGHT:
                player_points.append(cnt_pos)
    
    for cnt in red_contours:
        cnt_pos = getCntPosition(cnt)
        if len(cnt_pos) == 0: continue
        if cv2.contourArea(cnt) > 20:
            if cntIsMoving(cnt, gray_frame, prev_frame):
                player_points.append(cnt_pos)
            else:
                code_pos.append(cnt_pos)

    return gray_frame, cursor_pos, player_points, code_pos

def cntIsMoving(cnt, gray_frame, prev_frame) -> bool:
    # Get bounding box coordinates
    draw_x, draw_y, w, h = cv2.boundingRect(cnt)

    # Extract the region in the current frame for the object
    current_object_region = gray_frame[draw_y:draw_y+h, draw_x:draw_x+w]

    # Extract the region in the previous frame for the same object
    prev_object_region = prev_frame[draw_y:draw_y+h, draw_x:draw_x+w]

    # Calculate the absolute difference between the two regions
    region_diff = cv2.absdiff(current_object_region, prev_object_region)

    return np.sum(region_diff) > 2000

def getCntPosition(cnt):
    m = cv2.moments(cnt)
    if m["m00"] != 0:
        centroid_x = int(m["m10"] / m["m00"])
        centroid_y = int(m["m01"] / m["m00"])
        return np.array([centroid_x, centroid_y])
    return np.array([])

def normalize(vect):
    norm = np.linalg.norm(vect)
    if norm == 0: 
       return vect
    return vect / norm

if __name__ == "__main__":
    main()