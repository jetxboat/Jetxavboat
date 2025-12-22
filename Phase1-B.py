
"""
- PID steering for smooth control
- Recursive hunting when target lost
- Horizontal bottle orientation detection
- PS4 controller support
- Visual debugging
"""

import RPi.GPIO as GPIO
import pygame
import time
import cv2 # Handles camera input and displays detection frames.
import numpy as np # Numerical operations (centering, steering, detection math)
from ultralytics import YOLO # runs the YOLO model for object detection

# ====== GPIO CONFIGURATION ======
ENA, ENB = 15, 33  # PWM pins
IN1, IN2 = 36, 38  # Right motor
IN3, IN4 = 40, 37  # Left motor
PWM_FREQ = 100

# ====== CONTROL PARAMETERS ======
MAX_SPEED = 100
DEADZONE = 0.15
TURN_SENSITIVITY = 0.7

# ====== AI PARAMETERS ======
AI_BASE_SPEED = 80
AI_ATTACK_SPEED = 95
AI_CONFIDENCE_THRESHOLD = 0.6
BOTTLE_CLASS_ID = 0

# ====== PID PARAMETERS ======
# These control how smoothly the boat steers
KP = 0.5    # Proportional gain (main steering response)
KI = 0.01   # Integral gain (corrects steady-state error)
KD = 0.2    # Derivative gain (dampens oscillation)

# ====== CENTERING THRESHOLD ======
CENTER_THRESHOLD = 0.08  # How centered before ramming

# ====== SEARCH PARAMETERS ======
LOOK_AROUND_SPEED = 35      # Slower, smoother turns for camera
LOOK_WAIT_FRAMES = 80       # Wait ~2 seconds at each position (80 frames at 40fps)
LOOK_TURN_FRAMES = 40       # Turn duration ~1 second
SEARCH_SPIN_SPEED = 30      # Very gentle search spin

# Global variables
rpwm = None
lpwm = None
cam = None
model = None
current_mode = "MANUAL"

# PID variables
pid_integral = 0
pid_last_error = 0

# search variables
frames_without_target = 0
total_search_time = 0

# ====== GPIO SETUP ======
def setup_pins():
    """Initialize GPIO pins and PWM"""
    global rpwm, lpwm
    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    
    pins = (ENA, ENB, IN1, IN2, IN3, IN4)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
    
    rpwm = GPIO.PWM(ENA, PWM_FREQ)
    lpwm = GPIO.PWM(ENB, PWM_FREQ)
    rpwm.start(0)
    lpwm.start(0)
    
    print("GPIO initialized")

# ====== CAMERA SETUP ======
def setup_camera():
    """Initialize Webcam"""
    global cam
    
    for camera_index in [0, 1, 2]:
        cam = cv2.VideoCapture(camera_index)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            print(f"Webcam initialized (index: {camera_index})")
            return True
    
    print("Unable to open webcam")
    return False

# ====== AI MODEL SETUP ======
def setup_model():
    """Load YOLO model"""
    global model
    
    print("Loading YOLO model...")
    model = YOLO("best.pt")
    model.to("cuda")
    model.fuse()
    model.model.half()
    print("YOLO model loaded")

# ====== PS4 CONTROLLER ======
def init_ps4_controller():
    """Initialize PS4 controller"""
    pygame.init()
    pygame.joystick.init()
    
    joystick_count = pygame.joystick.get_count()
    
    if joystick_count == 0:
        print("No controller found - AI mode only")
        return None
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"Controller: {joystick.get_name()}")
    return joystick

# ====== MOTOR CONTROL ======
def set_motor_speed(right_speed, left_speed):
    """Set motor speeds with direction (-100 to +100)"""
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
    left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    
    # Right motor
    if right_speed >= 0:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        rpwm.ChangeDutyCycle(abs(right_speed))
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        rpwm.ChangeDutyCycle(abs(right_speed))
    
    # Left motor
    if left_speed >= 0:
        GPIO.output(IN4, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        lpwm.ChangeDutyCycle(abs(left_speed))
    else:
        GPIO.output(IN4, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        lpwm.ChangeDutyCycle(abs(left_speed))

def stop_motors():
    """Stop both motors"""
    rpwm.ChangeDutyCycle(0)
    lpwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def apply_deadzone(value, deadzone=DEADZONE):
    """Apply deadzone to joystick input"""
    if abs(value) < deadzone:
        return 0.0
    sign = 1 if value > 0 else -1
    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return sign * scaled

def arcade_drive(joystick):
    """Arcade drive mode from PS4 controller"""
    throttle = -joystick.get_axis(1)
    turn = joystick.get_axis(0)
    
    throttle = apply_deadzone(throttle)
    turn = apply_deadzone(turn)
    
    left_speed = throttle + (turn * TURN_SENSITIVITY)
    right_speed = throttle - (turn * TURN_SENSITIVITY)
    
    max_magnitude = max(abs(left_speed), abs(right_speed))
    if max_magnitude > 1.0:
        left_speed /= max_magnitude
        right_speed /= max_magnitude
    
    left_speed *= MAX_SPEED
    right_speed *= MAX_SPEED
    
    return right_speed, left_speed

# ====== BOTTLE DETECTION WITH ORIENTATION ======
def detect_bottle_with_orientation(results, frame_width):
    """
    Detect bottles and check if horizontal or vertical
    Returns: (detected, x_center, y_center, confidence, is_horizontal, width, height)
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id == BOTTLE_CLASS_ID and confidence > AI_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                
                # Calculate dimensions
                width = x2 - x1
                height = y2 - y1
                
                # Determine orientation
                # Horizontal bottle: width > height (floating on side)
                # Vertical bottle: height > width (standing up)
                is_horizontal = width > height * 1.2  # 20% margin
                
                return True, x_center, y_center, confidence, is_horizontal, width, height
    
    return False, None, None, None, False, None, None

# ====== PID STEERING CONTROLLER ======
def pid_steer_towards_bottle(bottle_x, frame_width, is_horizontal=False):
    """
    PID controller for smooth steering
    Adjusts for horizontal bottles (aim slightly above center)
    """
    global pid_integral, pid_last_error
    
    frame_center = frame_width / 2
    
    # If bottle is horizontal, we might want to aim for its center more precisely
    error = (bottle_x - frame_center) / frame_center  # Normalized -1 to +1
    
    # PID calculations
    pid_integral += error
    pid_integral = max(-1, min(1, pid_integral))  # Anti-windup
    
    derivative = error - pid_last_error
    pid_last_error = error
    
    # PID output
    correction = (KP * error) + (KI * pid_integral) + (KD * derivative)
    correction = max(-1, min(1, correction))  # Clamp to -1 to +1
    
    # Check if centered enough to attack
    if abs(error) < CENTER_THRESHOLD:
        # ATTACK MODE - Full speed ahead!
        pid_integral = 0  # Reset integral when attacking
        return AI_ATTACK_SPEED, AI_ATTACK_SPEED
    
    # Calculate differential steering
    # Correction > 0 means turn right (slow down right motor)
    # Correction < 0 means turn left (slow down left motor)
    right_speed = AI_BASE_SPEED - (correction * AI_BASE_SPEED * 0.8)
    left_speed = AI_BASE_SPEED + (correction * AI_BASE_SPEED * 0.8)
    
    # Clamp speeds
    right_speed = max(10, min(AI_BASE_SPEED, right_speed))
    left_speed = max(10, min(AI_BASE_SPEED, left_speed))
    
    return right_speed, left_speed

# ====== SMART HUNTING MODE ======
def smart_search():
    """
    slow search when target is lost:
    
    Phase 1 (0-80 frames / 2 sec): STOP and wait - look straight ahead
    Phase 2 (80-120 frames / 1 sec): Turn RIGHT slowly
    Phase 3 (120-200 frames / 2 sec): STOP and look right - wait for AI
    Phase 4 (200-240 frames / 1 sec): Turn back to CENTER slowly  
    Phase 5 (240-320 frames / 2 sec): STOP and look center - check again
    Phase 6 (320-360 frames / 1 sec): Turn LEFT slowly
    Phase 7 (360-440 frames / 2 sec): STOP and look left - wait for AI
    Phase 8 (440-480 frames / 1 sec): Turn back to CENTER
    Phase 9 (480-560 frames / 2 sec): Final check at center
    Phase 10 (560+ frames): Gentle 360° spin then STOP
    """
    global frames_without_target
    
    f = frames_without_target
    
    # Phase 1: Wait and look straight (2 seconds)
    if f < 80:
        stop_motors()
        progress = f / 80
        status = f"WAITING... ({progress*100:.0f}%)"
    
    # Phase 2: Turn RIGHT slowly (1 second)
    elif f < 120:
        set_motor_speed(LOOK_AROUND_SPEED, -LOOK_AROUND_SPEED)
        status = "TURNING RIGHT..."
    
    # Phase 3: STOP and look right (2 seconds - AI detection time)
    elif f < 200:
        stop_motors()
        progress = (f - 120) / 80
        status = f"LOOKING RIGHT ({progress*100:.0f}%)"
    
    # Phase 4: Turn back to CENTER (1 second)
    elif f < 240:
        set_motor_speed(-LOOK_AROUND_SPEED, LOOK_AROUND_SPEED)
        status = "RETURNING TO CENTER..."
    
    # Phase 5: STOP at center and check (2 seconds)
    elif f < 320:
        stop_motors()
        progress = (f - 240) / 80
        status = f"CHECKING CENTER ({progress*100:.0f}%)"
    
    # Phase 6: Turn LEFT slowly (1 second)
    elif f < 360:
        set_motor_speed(-LOOK_AROUND_SPEED, LOOK_AROUND_SPEED)
        status = "TURNING LEFT..."
    
    # Phase 7: STOP and look left (2 seconds - AI detection time)
    elif f < 440:
        stop_motors()
        progress = (f - 360) / 80
        status = f"LOOKING LEFT ({progress*100:.0f}%)"
    
    # Phase 8: Turn back to CENTER (1 second)
    elif f < 480:
        set_motor_speed(LOOK_AROUND_SPEED, -LOOK_AROUND_SPEED)
        status = "RETURNING TO CENTER..."
    
    # Phase 9: Final center check (2 seconds)
    elif f < 560:
        stop_motors()
        progress = (f - 480) / 80
        status = f"FINAL CHECK ({progress*100:.0f}%)"
    
    # Phase 10: Nothing found - very slow 360 spin for 5 seconds, then stop
    elif f < 760:
        set_motor_speed(SEARCH_SPIN_SPEED, -SEARCH_SPIN_SPEED)
        status = "SLOW 360° SEARCH..."
    
    else:
        # Give up - stop completely and wait
        stop_motors()
        status = "NO TARGETS - Stopped"
    
    return status

# ====== MAIN PROGRAM ======
def main():
    global current_mode, frames_without_target, total_search_time
    global pid_integral, pid_last_error
    
    print("=" * 60)
    print("ULTIMATE AUTONOMOUS BOTTLE HUNTING BOAT")
    print("=" * 60)
    
    # Setup
    setup_pins()
    joystick = init_ps4_controller()
    
    if not setup_camera():
        print("Camera required")
        GPIO.cleanup()
        return
    
    setup_model()
    
    print("\n CONTROLS:")
    print("   Left Stick: Drive (Manual Mode)")
    print("   Triangle: Toggle AI/Manual Mode")
    print("   Circle : Emergency Stop")
    print("   Options: Quit")
    print("\n Starting in MANUAL mode")
    print("=" * 60 + "\n")
    
    # Main loop
    clock = pygame.time.Clock()
    frame_width = 800
    frame_height = 600
    running = True
    
    try:
        while running:
            # Get camera frame
            ret, frame = cam.read()
            if not ret:
                print(" Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Check controller buttons
            if joystick:
                # Square - Toggle mode
                if joystick.get_button(2):
                    if current_mode == "MANUAL":
                        current_mode = "AUTONOMOUS"
                        frames_without_target = 0
                        pid_integral = 0
                        pid_last_error = 0
                        print("\n AUTONOMOUS MODE - AI HUNTING!")
                    else:
                        current_mode = "MANUAL"
                        print("\n MANUAL MODE")
                    time.sleep(0.3)
                
                # Circle - Emergency stop
                if joystick.get_button(1):
                    print("\n EMERGENCY STOP")
                    stop_motors()
                    time.sleep(0.1)
                    continue
                
                # Options - Quit
                if joystick.get_button(9):
                    print("\n Quitting...")
                    running = False
                    break
            
            # ====== MANUAL MODE ======
            if current_mode == "MANUAL":
                if joystick:
                    right_speed, left_speed = arcade_drive(joystick)
                    set_motor_speed(right_speed, left_speed)
                    
                    if abs(right_speed) > 1 or abs(left_speed) > 1:
                        print(f"MANUAL | R: {right_speed:+6.1f}% | L: {left_speed:+6.1f}%", end='\r')
                else:
                    stop_motors()
                
                cv2.putText(frame, "MANUAL MODE - Press Square for AI", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Boat Camera", frame)
            
            # ====== AUTONOMOUS MODE ======
            else:
                # Run YOLO detection
                results = model(frame, conf=AI_CONFIDENCE_THRESHOLD, verbose=False)
                annotated = results[0].plot()
                
                # Detect bottles with orientation
                detected, bottle_x, bottle_y, confidence, is_horizontal, width, height = \
                    detect_bottle_with_orientation(results, frame_width)
                
                if detected:
                    # TARGET FOUND!
                    frames_without_target = 0
                    
                    # Calculate error for display
                    frame_center = frame_width / 2
                    error = bottle_x - frame_center
                    normalized_error = error / frame_center
                    
                    # Calculate steering with PID
                    right_speed, left_speed = pid_steer_towards_bottle(bottle_x, frame_width, is_horizontal)
                    set_motor_speed(right_speed, left_speed)
                    
                    # Visual feedback
                    off_center_amount = abs(normalized_error)
                    
                    # Draw reference lines
                    cv2.line(annotated, (int(frame_center), 0), (int(frame_center), frame_height), 
                            (255, 255, 0), 2)  # Yellow center line
                    cv2.line(annotated, (int(bottle_x), 0), (int(bottle_x), frame_height), 
                            (0, 255, 0), 3)  # Green bottle line
                    
                    # Draw bottle bounding box with orientation indicator
                    if is_horizontal:
                        orientation_text = "HORIZONTAL"
                        orientation_color = (255, 165, 0)  # Orange
                    else:
                        orientation_text = "VERTICAL"
                        orientation_color = (0, 255, 255)  # Cyan
                    
                    # Status text
                    if off_center_amount < CENTER_THRESHOLD:
                        status = "LOCKED! RAMMING!"
                        color = (0, 0, 255)
                    elif off_center_amount < 0.2:
                        status = f"PID STEERING | {(1-off_center_amount)*100:.0f}%"
                        color = (0, 255, 255)
                    else:
                        status = "TURNING TO TARGET"
                        color = (255, 165, 0)
                    
                    # Direction indicator
                    if normalized_error > 0.05:
                        direction = "TURN RIGHT"
                    elif normalized_error < -0.05:
                        direction = "TURN LEFT"
                    else:
                        direction = "CENTERED"
                    
                    print(f"{status} | {orientation_text} | R: {right_speed:.0f}% L: {left_speed:.0f}%", end='\r')
                    
                    # Display info
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(annotated, direction, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated, orientation_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, orientation_color, 2)
                    cv2.putText(annotated, f"Conf: {confidence:.2f}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                else:
                    # TARGET LOST - Use smart search (no memory)
                    frames_without_target += 1
                    search_status = smart_search()
                    
                    print(f"{search_status} | Lost for {frames_without_target} frames", end='\r')
                    
                    cv2.putText(annotated, f"{search_status}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                    cv2.putText(annotated, f"Lost: {frames_without_target} frames ({frames_without_target/40:.1f}s)", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(annotated, "Press Square for Manual", 
                           (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Boat Camera", annotated)
            
            # Handle window events
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            
            # Maintain loop rate
            clock.tick(40)  # 40 FPS
    
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    
    finally:
        print("\n Cleaning up...")
        stop_motors()
        rpwm.stop()
        lpwm.stop()
        if cam:
            cam.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        if joystick:
            pygame.quit()
        print("Cleanup complete! Happy hunting! ")

if __name__ == "__main__":
    main()