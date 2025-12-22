"""
CATAMARAN BLDC BOAT CONTROLLER
Complete and tested version

3 modes:
1. PS4 Advanced (R2=speed, left stick=differential control)
2. AI Simple + R2 Speed + Stick Differential
3. AI Smart Search + R2 Speed + Stick Differential
"""

import time
import board
import pygame
import cv2
import numpy as np
import busio # I2C communication
from adafruit_pca9685 import PCA9685 # Control ESCs through PCA9685 PWM driver
from ultralytics import YOLO

# ====== MOTOR CONFIGURATION ======
# PCA9685 Freq, ESCs usually use 50Hz
FREQUENCY = 50 # Hz

# PCA9685 PWM Channels
MOTOR_L_CHANNEL = 0 
MOTOR_R_CHANNEL = 1
# Servo : SG-5010 
SERVO1_CHANNEL = 2


# DISCOVERED VALUES 
# Microsecond pulse values for ESC control
MOTOR_L_MIN = 3900 # Minimum throttle
MOTOR_L_MAX = 5500 # Maximum throttle

MOTOR_R_MIN = 3900
MOTOR_R_MAX = 5500

SERVO1_MIN = 1800   # Right position
SERVO1_CENTER = 4000  # Center
SERVO1_MAX = 8400   # Left position

# SAFETY LIMIT
MOTOR_SAFE_MAX = 4900

# START LIMIT (for initial testing)
INITIAL_MAX_SPEED = 80

# ====== NAVIGATION SPEEDS ======
FORWARD_SPEED = 60  # %
TURN_SPEED = 0     # %

# Motor trim for balancing
MOTOR_L_TRIM = 1.0
MOTOR_R_TRIM = 1.0

# ====== PS4 SETTINGS ======
DEADZONE = 0.15
R2_SENSITIVITY = 1.3

# ====== AI PARAMETERS ======
MODEL = "best.pt"  # YOLO model path
AI_APPROACH_SPEED = 40          
AI_CLOSE_APPROACH_SPEED = 30    
AI_CENTERING_THRESHOLD = 0.05
AI_CLOSE_DISTANCE = 0.4
AI_CONFIDENCE_THRESHOLD = 0.55
BOTTLE_CLASS_ID = 0
BOTTLE_CLASS_ID1 = 1

# ====== PID PARAMETERS ======
KP = 1     # 0.3    
KI = 0.01  # 0.005  
KD = 0.2   # 0.15   

# ====== SEARCH PARAMETERS ======                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
LOOK_TURN_SPEED = 30           
LOOK_TURN_FRAMES_45 =  25
LOOK_WAIT_FRAMES = 80
SEARCH_SPIN_SPEED = 40

# ====== SMOOTHING ======
ACCELERATION_RATE = 2.0  # Max % change per frame
DECELERATION_RATE = 3.0  # Faster deceleration for safety

# Smooth speed tracking
current_l_speed = 0.0
current_r_speed = 0.0

# ====== DISPLAY ======
SHOW_CAMERA = False  # Set False for SSH

print("=" * 70)
print(" CATAMARAN BLDC CONTROLLER - FIXED VERSION")
print("=" * 70)

# Globals
pca = None
cam = None
model = None
ai_enabled = False
pid_integral = 0
pid_last_error = 0
frames_without_target = 0

# ====== SETUP FUNCTIONS ======
def setup_pca():
    global pca
    print("\n[1/4] Initializing PCA9685...")
    i2c = busio.I2C(board.SCL_1, board.SDA_1)
    pca = PCA9685(i2c, address=0x40)
    pca.frequency = FREQUENCY
    print("   ✓ PCA9685 ready")

def setup_camera():
    global cam
    print("[2/4] Initializing camera...")
    for idx in [0, 1, 2]:
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            print(f"    ✓ Camera ready (index {idx})")
            return True
    print("   ✗ No camera found")
    return False

def setup_model():
    global model
    print("[3/4] Loading YOLO model...")
    try:
        model = YOLO(MODEL)
        model.to("cuda")
        model.fuse()
        model.model.half()
        print("      ✓ Model loaded\n")
    except Exception as e:
        print(f"      ✗ Model load failed: {e}")
        print("      Falling back to CPU inference (slower)")
        try:
            model = YOLO("best.pt")
            print("      ✓ Model loaded on CPU\n")
        except Exception as e2:
            print(f"      ✗ Fatal: Cannot load model: {e2}")
            model = None

def init_ps4():
    print("[4/4] Initializing PS4 controller...")
    pygame.init()
    pygame.joystick.init()
    
    # Retry detection (sometimes pygame needs time to find joystick)
    for attempt in range(3):
        if pygame.joystick.get_count() > 0:
            joy = pygame.joystick.Joystick(0)
            joy.init()
            print(f"✓ Controller: {joy.get_name()}\n")
            return joy
        
        if attempt < 2:
            print(f"  [Attempt {attempt + 1}/3] Waiting for controller...")
            time.sleep(0.5)
    
    print("✗ No PS4 controller detected")
    print("  Make sure: jstest /dev/input/js0 works first")
    print("  Then restart this script\n")
    return None

# ====== MOTOR CONTROLLER ======
class MotorController:
    def __init__(self): # SET PCA9685 CHANNELS
        self.l_ch = pca.channels[MOTOR_L_CHANNEL]
        self.r_ch = pca.channels[MOTOR_R_CHANNEL]
        
        # Calculate safe percentages
        self.l_safe_pct = ((MOTOR_SAFE_MAX - MOTOR_L_MIN) / (MOTOR_L_MAX - MOTOR_L_MIN)) * 100
        self.r_safe_pct = ((MOTOR_SAFE_MAX - MOTOR_R_MIN) / (MOTOR_R_MAX - MOTOR_R_MIN)) * 100
        
        print(f"Motor Limits:")
        print(f"  Left:  {MOTOR_L_MIN}-{MOTOR_SAFE_MAX}µs ({self.l_safe_pct:.0f}% max)")
        print(f"  Right: {MOTOR_R_MIN}-{MOTOR_SAFE_MAX}µs ({self.r_safe_pct:.0f}% max)")
        print(f"  Initial speed limit: {INITIAL_MAX_SPEED}%\n")
    
    def percent_to_pulse(self, pct_l, pct_r):
        # Apply trim
        pct_l *= MOTOR_L_TRIM
        pct_r *= MOTOR_R_TRIM
        
        # Apply safety limits
        pct_l = max(0, min(self.l_safe_pct, pct_l))
        pct_r = max(0, min(self.r_safe_pct, pct_r))
        
        # Linear mapping: % → µs
        pulse_l = MOTOR_L_MIN + (MOTOR_L_MAX - MOTOR_L_MIN) * pct_l / 100
        pulse_r = MOTOR_R_MIN + (MOTOR_R_MAX - MOTOR_R_MIN) * pct_r / 100
        
        # Hard clamp
        pulse_l = min(MOTOR_SAFE_MAX, int(pulse_l))
        pulse_r = min(MOTOR_SAFE_MAX, int(pulse_r))
        
        return pulse_l, pulse_r
    
    def set_motors_percent(self, pct_l, pct_r): 
        # Convert % to pulse, then set
        pulse_l, pulse_r = self.percent_to_pulse(pct_l, pct_r)
        # Set duty cycle directly (0-65535 represents 0-20ms)
        self.l_ch.duty_cycle = pulse_l
        self.r_ch.duty_cycle = pulse_r
        return pulse_l, pulse_r
    
    def set_motors_pulse(self, pulse_l, pulse_r):
        pulse_l = max(MOTOR_L_MIN, min(MOTOR_SAFE_MAX, int(pulse_l)))
        pulse_r = max(MOTOR_R_MIN, min(MOTOR_SAFE_MAX, int(pulse_r)))
        # Set duty cycle directly
        self.l_ch.duty_cycle = pulse_l
        self.r_ch.duty_cycle = pulse_r
        return pulse_l, pulse_r
    
    def arm_motors(self):
        print("\n  Arming ESCs...")
        print("   Setting minimum throttle...")
        
        # Set to minimum pulse directly
        self.l_ch.duty_cycle = MOTOR_L_MIN
        self.r_ch.duty_cycle = MOTOR_R_MIN
        
        print(f"   Left:  {MOTOR_L_MIN}µs")
        print(f"   Right: {MOTOR_R_MIN}µs")
        
        print("   Waiting for beep sequence...")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("   ✓ Motors armed and ready!\n")
    
    def stop(self):
        self.l_ch.duty_cycle = MOTOR_L_MIN
        self.r_ch.duty_cycle = MOTOR_R_MIN
    
    def cleanup(self):
        self.stop()
        time.sleep(0.5)
        self.l_ch.duty_cycle = 0
        self.r_ch.duty_cycle = 0

# ====== SERVO CONTROLLER ======
class ServoController:
    def __init__(self):
        self.s1_min = SERVO1_MIN
        self.s1_center = SERVO1_CENTER
        self.s1_max = SERVO1_MAX
        
        self.servo1 = pca.channels[SERVO1_CHANNEL]
        
        print(f"Servo 1 (SG-5010) on channel {SERVO1_CHANNEL}")
        print(f"  Pulse range: {self.s1_min}-{self.s1_max}µs")
        print(f"  Center pulse: {self.s1_center}µs\n")

    def set_servo1_pulse(self, pulse_us):
        pulse_us = max(0, min(65535, pulse_us))
        self.servo1.duty_cycle = int(pulse_us)
        return pulse_us
    
    def set_servo1_percent(self, percent):
        percent = max(0, min(100, percent))
        pulse_range = self.s1_max - self.s1_min
        pulse = self.s1_min + (pulse_range * percent / 100)
        self.set_servo1_pulse(int(pulse))
        return int(pulse)
    
    def open_net(self):
        # Center 
        self.set_servo1_percent(50)
    
    def close_net(self):
        # Closed
        self.set_servo1_percent(0)
    
    def stop(self):
        # Stop PWM signal to servo
        self.servo1.duty_cycle = 0
    
    def cleanup(self):
        # Safe cleanup
        print("\n Cleaning up servo...")
        self.stop()
        time.sleep(0.5)

# ====== HELPER FUNCTIONS ======
def apply_deadzone(val, dz=DEADZONE):
    if abs(val) < dz:
        return 0.0
    sign = 1 if val > 0 else -1
    scaled = (abs(val) - dz) / (1.0 - dz)
    return sign * scaled

def smooth_speed_change(current, target, accel_rate, decel_rate):
    # Smooth acceleration/deceleration
    diff = target - current
    
    if abs(diff) < 0.5:
        return target
    
    if diff > 0:
        # Accelerating
        return current + min(accel_rate, diff)
    else:
        # Decelerating
        return current + max(-decel_rate, diff)

def smooth_turn(base_speed, turn_amt):
    # Reduce one motor speed for turning (no increase for safety)
    if turn_amt > 0:  # Turn right, reduce right motor
        l_speed = base_speed
        r_speed = base_speed * (1 - abs(turn_amt) * 0.7)  
    else:  # Turn left, reduce left motor
        l_speed = base_speed * (1 - abs(turn_amt) * 0.7)
        r_speed = base_speed
    
    return l_speed, r_speed

# ====== BOTTLE DETECTION ======
def detect_bottle(results, fw, fh):
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            
            if conf > AI_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Calculate center
                x_cen = (x1 + x2) / 2
                y_cen = (y1 + y2) / 2
                
                # Calculate size
                bbox_area = (x2 - x1) * (y2 - y1)
                area_ratio = bbox_area / (fw * fh)
                
                return True, x_cen, y_cen, conf, area_ratio
    
    return False, None, None, None, None

# ====== PID STEERING ======
def pid_steer_to_bottle(bottle_x, fw, area_ratio):
    global pid_integral, pid_last_error
    
    fc = fw / 2
    error = (bottle_x - fc) / fc
    
    pid_integral += error
    pid_integral = max(-1, min(1, pid_integral))
    
    deriv = error - pid_last_error
    pid_last_error = error
    
    correction = (KP * error) + (KI * pid_integral) + (KD * deriv)
    correction = max(-1, min(1, correction))
    
    # Speed based on distance
    if area_ratio > AI_CLOSE_DISTANCE:
        base = AI_CLOSE_APPROACH_SPEED
        status = " SLOW"
    else:
        base = AI_APPROACH_SPEED
        status = " APPROACH"
    
    if abs(error) < AI_CENTERING_THRESHOLD:
        l_spd = base
        r_spd = base
        status = " CENTERED - " + status
    else:
        l_spd, r_spd = smooth_turn(base, correction)
        if correction > 0:
            status = " RIGHT - " + status
        else:
            status = " LEFT - " + status
    
    return l_spd, r_spd, status

# ====== SMART SEARCH ======
def smart_search():
    global frames_without_target
    f = frames_without_target
    
    # Phase timings 
    p1_end = LOOK_WAIT_FRAMES
    p2_end = p1_end + LOOK_TURN_FRAMES_45
    p3_end = p2_end + LOOK_WAIT_FRAMES
    p4_end = p3_end + LOOK_TURN_FRAMES_45
    p5_end = p4_end + LOOK_WAIT_FRAMES
    p6_end = p5_end + LOOK_TURN_FRAMES_45
    p7_end = p6_end + LOOK_WAIT_FRAMES
    p8_end = p7_end + LOOK_TURN_FRAMES_45
    p9_end = p8_end + LOOK_WAIT_FRAMES
    p10_end = p9_end + 200
    
    if f < p1_end:
        return 0, 0, f" CENTER ({f}/{LOOK_WAIT_FRAMES})"
    
    elif f < p2_end:
        # OPTION A: One motor only
        return LOOK_TURN_SPEED, 0, " TURN RIGHT 45°"
        
        # OPTION B: Counter-rotating
        # return LOOK_TURN_SPEED, -LOOK_TURN_SPEED, "TURN RIGHT 45°"
    
    elif f < p3_end:
        return 0, 0, f" LOOK RIGHT ({f-p2_end}/{LOOK_WAIT_FRAMES})"
    
    elif f < p4_end:
        # OPTION A: One motor only
        return 0, LOOK_TURN_SPEED, " RETURN CENTER"
        
        # OPTION B: Counter-rotating
        # return -LOOK_TURN_SPEED, LOOK_TURN_SPEED, "RETURN CENTER"
    
    elif f < p5_end:
        return 0, 0, f" CENTER ({f-p4_end}/{LOOK_WAIT_FRAMES})"
    
    elif f < p6_end:
        # OPTION A: One motor only
        return 0, LOOK_TURN_SPEED, " TURN LEFT 45°"
        
        # OPTION B: Counter-rotating
        # return -LOOK_TURN_SPEED, LOOK_TURN_SPEED, "↺ TURN LEFT 45°"
    
    elif f < p7_end:
        return 0, 0, f" LOOK LEFT ({f-p6_end}/{LOOK_WAIT_FRAMES})"
    
    elif f < p8_end:
        # OPTION A: One motor only (current)
        return LOOK_TURN_SPEED, 0, " RETURN CENTER"
        
        # OPTION B: Counter-rotating
        # return LOOK_TURN_SPEED, -LOOK_TURN_SPEED, " RETURN CENTER"
    
    elif f < p9_end:
        return 0, 0, f" FINAL CHECK ({f-p8_end}/{LOOK_WAIT_FRAMES})"
    
    elif f < p10_end:
        return SEARCH_SPIN_SPEED, 0, " SPIN 360°"
    
        # 360° search - counter-rotating
        # return SEARCH_SPIN_SPEED, -SEARCH_SPIN_SPEED, " 360° SEARCH"
    
    # else:
       # return 0, 0, " STOPPED - NO TARGET"
    
    else:
        frames_without_target = 0  # Reset to loop search
        return 0, 0, " SEARCH RESTART"

# ====== CONTROL MODES ======
def mode_ps4_advanced(motors,joy):
    print("\n" + "=" * 70)
    print("MODE 1: PS4 ADVANCED - R2 SPEED + STICK DIFFERENTIAL")
    print("=" * 70)
    print(f"\n R2 Trigger: Master speed (0-{INITIAL_MAX_SPEED}%)")
    print("   Left Stick X: -1.0 (all left) to +1.0 (all right)")
    print("   Left Stick X=0: Both motors equal")
    print("   Circle: Stop | Options: Quit\n")
    
    global current_l_speed, current_r_speed
    current_l_speed = 0
    current_r_speed = 0
    
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # R2 controls master speed (0-100%, scaled to INITIAL_MAX_SPEED)
            r2_raw = (joy.get_axis(5) + 1) / 2  # -1 to 1 → 0 to 1
            r2_raw = min(1.0, r2_raw * R2_SENSITIVITY)  # Apply sensitivity
            master_speed = r2_raw * INITIAL_MAX_SPEED
            
            # Left stick X controls differential (-1 to 1)
            stick_x = apply_deadzone(joy.get_axis(0))
            
            # Calculate motor speeds based on stick position
            if abs(stick_x) < 0.05:
                # Center: both motors equal
                target_l = master_speed
                target_r = master_speed
                blend_status = "BOTH"
            elif stick_x > 0:
                # Right: left motor full, right motor reduced
                target_l = master_speed
                target_r = master_speed * (1 - stick_x)  # 0 to 100% right = 100% to 0%
                blend_status = f"R {stick_x*100:.0f}%"
            else:
                # Left: right motor full, left motor reduced
                target_r = master_speed
                target_l = master_speed * (1 + stick_x)  # -100% to 0% left = 0% to 100%
                blend_status = f"L {-stick_x*100:.0f}%"
            
            # Smooth acceleration
            current_l_speed = smooth_speed_change(current_l_speed, target_l,
                                                  ACCELERATION_RATE, DECELERATION_RATE)
            current_r_speed = smooth_speed_change(current_r_speed, target_r,
                                                  ACCELERATION_RATE, DECELERATION_RATE)
            
            motors.set_motors_percent(current_l_speed, current_r_speed)
            
            if joy.get_button(1):  # Circle - Emergency stop
                current_l_speed = 0
                current_r_speed = 0
                motors.stop()
                print("\n STOP")
                time.sleep(0.3)
            
            if joy.get_button(9):  # Options - Quit
                running = False
            
            if master_speed > 0.5:
                print(f" Speed: {master_speed:.1f}% | Blend: {blend_status} | L:{current_l_speed:.0f}% R:{current_r_speed:.0f}%     ", end='\r')
            
            clock.tick(50)
    
    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        motors.stop()

def mode_ai_simple_advanced(motors, servo, joy):
    print("\n" + "=" * 70)
    print("MODE 2: AI SIMPLE + R2 SPEED + STICK DIFFERENTIAL")
    print("=" * 70)
    print(f"\n Triangle: Toggle AI ON/OFF")
    print(f" R2 Trigger: Master speed (0-{INITIAL_MAX_SPEED}%)")
    print("   Left Stick X: Differential control (-1 left to +1 right)")
    print("   Circle: Emergency Stop | Options: Quit")
    print("\n Starting in MANUAL (press Triangle for AI)\n")
    
    global ai_enabled, pid_integral, pid_last_error, frames_without_target
    global current_l_speed, current_r_speed
    
    ai_enabled = False
    pid_integral = 0
    pid_last_error = 0
    frames_without_target = 0
    current_l_speed = 0
    current_r_speed = 0
    
    clock = pygame.time.Clock()
    fw, fh = 800, 600
    running = True
    
    try:
        while running:
            ret, frame = cam.read()
            if not ret:
                print("\n Camera frame dropped", end='\r')
                time.sleep(0.1)
                continue
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Triangle - Toggle AI
            if joy.get_button(2):
                ai_enabled = not ai_enabled
                if ai_enabled:
                    print("\n AI ENABLED - Simple chase (no search)")
                else:
                    print("\n MANUAL MODE")
                    motors.stop()
                frames_without_target = 0
                pid_integral = 0
                pid_last_error = 0
                current_l_speed = 0
                current_r_speed = 0
                time.sleep(0.3)
            
            # Circle - Emergency stop
            if joy.get_button(1):
                motors.stop()
                current_l_speed = 0
                current_r_speed = 0
                print("\n EMERGENCY STOP")
                time.sleep(0.3)
                continue
            
            # Options - Quit
            if joy.get_button(9):
                running = False
                break
            
            if ai_enabled:
                # AI MODE
                results = model(frame, conf=AI_CONFIDENCE_THRESHOLD, verbose=False)
                annotated = results[0].plot()
                
                detected, bx, by, conf, area = detect_bottle(results, fw, fh)
                
                if detected:
                    # BOTTLE DETECTED - All visual code goes HERE
                    frames_without_target = 0
                    servo.open_net()
                    
                    target_l, target_r, status = pid_steer_to_bottle(bx, fw, area)
                    
                    # Smooth speed changes
                    current_l_speed = smooth_speed_change(current_l_speed, target_l, ACCELERATION_RATE, DECELERATION_RATE)
                    current_r_speed = smooth_speed_change(current_r_speed, target_r, ACCELERATION_RATE, DECELERATION_RATE)
                    
                    motors.set_motors_percent(current_l_speed, current_r_speed)
                    
                    # Draw visuals (bx, by are valid here)
                    # Frame center line (yellow)
                    cv2.line(annotated, (int(fw/2), 0), (int(fw/2), fh), (255, 255, 0), 2)
                    # Bottle center line (green)
                    cv2.line(annotated, (int(bx), 0), (int(bx), fh), (0, 255, 0), 3)
                    # Status text
                    cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Confidence
                    cv2.putText(annotated, f"Conf: {conf:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # Center crosshair on bottle
                    cv2.line(annotated, (int(bx) - 20, int(by)), (int(bx) + 20, int(by)), (255, 0, 255), 2)
                    cv2.line(annotated, (int(bx), int(by) - 20), (int(bx), int(by) + 20), (255, 0, 255), 2)
                    # Error line (white)
                    cv2.line(annotated, (int(fw/2), int(fh/2)), (int(bx), int(fh/2)), (255, 255, 255), 2)
                    
                    print(f"{status} | L:{current_l_speed:.0f}% R:{current_r_speed:.0f}%", end='\r')
                
                else:
                    # NO BOTTLE - Different visuals
                    frames_without_target += 1
                    servo.close_net()
                    
                    current_l_speed = smooth_speed_change(current_l_speed, 0, ACCELERATION_RATE, DECELERATION_RATE)
                    current_r_speed = smooth_speed_change(current_r_speed, 0, ACCELERATION_RATE, DECELERATION_RATE)
                    motors.set_motors_percent(current_l_speed, current_r_speed)
                    
                    # Draw only center line (no bottle coordinates)
                    cv2.line(annotated, (int(fw/2), 0), (int(fw/2), fh), (255, 255, 0), 2)
                    cv2.putText(annotated, " NO TARGET - WAITING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    
                    print(f" NO TARGET ({frames_without_target} frames)", end='\r')
                
                # Common overlay (always visible)
                cv2.putText(annotated, "Triangle=Manual", (10, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if SHOW_CAMERA:
                    cv2.imshow("Catamaran", annotated)
            
            else:
                # MANUAL MODE - R2 Speed + Stick Differential
                r2_raw = (joy.get_axis(5) + 1) / 2
                r2_raw = min(1.0, r2_raw * R2_SENSITIVITY)
                master_speed = r2_raw * INITIAL_MAX_SPEED
                
                stick_x = apply_deadzone(joy.get_axis(0))
                
                if abs(stick_x) < 0.05:
                    target_l = master_speed
                    target_r = master_speed
                    blend_status = "BOTH"
                elif stick_x > 0:
                    target_l = master_speed
                    target_r = master_speed * (1 - stick_x)
                    blend_status = f"R {stick_x*100:.0f}%"
                else:
                    target_r = master_speed
                    target_l = master_speed * (1 + stick_x)
                    blend_status = f"L {-stick_x*100:.0f}%"
                
                current_l_speed = smooth_speed_change(current_l_speed, target_l, ACCELERATION_RATE, DECELERATION_RATE)
                current_r_speed = smooth_speed_change(current_r_speed, target_r, ACCELERATION_RATE, DECELERATION_RATE)
                
                motors.set_motors_percent(current_l_speed, current_r_speed)
                
                if master_speed > 0.5:
                    print(f" MANUAL | Speed: {master_speed:.1f}% | Blend: {blend_status} | L:{current_l_speed:.0f}% R:{current_r_speed:.0f}%", end='\r')
                
                # Draw center line
                cv2.line(frame, (int(fw/2), 0), (int(fw/2), fh), (0, 255, 255), 2)
                cv2.putText(frame, "MANUAL - Triangle=AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if SHOW_CAMERA:
                    cv2.imshow("Catamaran", frame)
            
            if SHOW_CAMERA and cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            
            clock.tick(40)
    
    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        motors.stop()
        if SHOW_CAMERA:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def mode_ai_search_advanced(motors, servo, joy):
    print("\n" + "=" * 70)
    print("MODE 3: AI SMART SEARCH + R2 SPEED + STICK DIFFERENTIAL")
    print("=" * 70)
    print(f"\n Triangle: Toggle AI ON/OFF")
    print(f" R2 Trigger: Master speed (0-{INITIAL_MAX_SPEED}%)")
    print("   Left Stick X: Differential control (-1 left to +1 right)")
    print("   Circle: Emergency Stop | Options: Quit")
    print("\n Starting in MANUAL (press Triangle for AI)\n")
    
    global ai_enabled, pid_integral, pid_last_error, frames_without_target
    global current_l_speed, current_r_speed
    
    ai_enabled = False
    pid_integral = 0
    pid_last_error = 0
    frames_without_target = 0
    current_l_speed = 0
    current_r_speed = 0
    
    clock = pygame.time.Clock()
    fw, fh = 800, 600
    running = True
    
    try:
        while running:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Triangle - Toggle AI
            if joy.get_button(2):
                ai_enabled = not ai_enabled
                if ai_enabled:
                    print("\n AI ENABLED - Smart search pattern")
                else:
                    print("\n MANUAL MODE")
                    motors.stop()
                frames_without_target = 0
                pid_integral = 0
                pid_last_error = 0
                current_l_speed = 0
                current_r_speed = 0
                time.sleep(0.3)
            
            # Circle - Emergency stop
            if joy.get_button(1):
                motors.stop()
                current_l_speed = 0
                current_r_speed = 0
                print("\n EMERGENCY STOP")
                time.sleep(0.3)
                continue
            
            # Options - Quit
            if joy.get_button(9):
                running = False
                break
            
            if ai_enabled:
                # AI MODE WITH SEARCH
                results = model(frame, conf=AI_CONFIDENCE_THRESHOLD, verbose=False)
                annotated = results[0].plot()
                
                detected, bx, by, conf, area = detect_bottle(results, fw, fh)
                
                if detected:
                    # TARGET FOUND - Draw visuals
                    frames_without_target = 0
                    servo.open_net()
                    
                    target_l, target_r, status = pid_steer_to_bottle(bx, fw, area)
                    
                    # Smooth speed changes
                    current_l_speed = smooth_speed_change(current_l_speed, target_l, ACCELERATION_RATE, DECELERATION_RATE)
                    current_r_speed = smooth_speed_change(current_r_speed, target_r, ACCELERATION_RATE, DECELERATION_RATE)
                    
                    motors.set_motors_percent(current_l_speed, current_r_speed)
                    
                    # Draw visuals (bx, by valid here)
                    # Frame center line (yellow)
                    cv2.line(annotated, (int(fw/2), 0), (int(fw/2), fh), (255, 255, 0), 2)
                    # Bottle center line (green)
                    cv2.line(annotated, (int(bx), 0), (int(bx), fh), (0, 255, 0), 3)
                    # Status
                    cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Confidence
                    cv2.putText(annotated, f"Conf: {conf:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # Center crosshair on bottle
                    cv2.line(annotated, (int(bx) - 20, int(by)), (int(bx) + 20, int(by)), (255, 0, 255), 2)
                    cv2.line(annotated, (int(bx), int(by) - 20), (int(bx), int(by) + 20), (255, 0, 255), 2)
                    # Error line
                    cv2.line(annotated, (int(fw/2), int(fh/2)), (int(bx), int(fh/2)), (255, 255, 255), 2)
                    
                    print(f"{status} | L:{current_l_speed:.0f}% R:{current_r_speed:.0f}%", end='\r')
                
                else:
                    # TARGET LOST - SEARCH PATTERN (no bottle coordinates)
                    frames_without_target += 1
                    servo.close_net()
                    
                    search_l, search_r, search_status = smart_search()
                    
                    # Smooth speed changes for search movements
                    current_l_speed = smooth_speed_change(current_l_speed, search_l, ACCELERATION_RATE/2, DECELERATION_RATE)
                    current_r_speed = smooth_speed_change(current_r_speed, search_r, ACCELERATION_RATE/2, DECELERATION_RATE)
                    
                    motors.set_motors_percent(current_l_speed, current_r_speed)
                    
                    # Draw only center line (no bottle)
                    cv2.line(annotated, (int(fw/2), 0), (int(fw/2), fh), (255, 255, 0), 2)
                    cv2.putText(annotated, search_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    cv2.putText(annotated, f"Lost: {frames_without_target} frames ({frames_without_target/40:.1f}s)",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    print(f"{search_status} | L:{current_l_speed:.0f}% R:{current_r_speed:.0f}%", end='\r')
                
                # Common overlay
                cv2.putText(annotated, "Triangle=Manual", (10, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if SHOW_CAMERA:
                    cv2.imshow("Catamaran", annotated)
            
            else:
                # MANUAL MODE - R2 Speed + Stick Differential
                r2_raw = (joy.get_axis(5) + 1) / 2
                r2_raw = min(1.0, r2_raw * R2_SENSITIVITY)
                master_speed = r2_raw * INITIAL_MAX_SPEED
                
                stick_x = apply_deadzone(joy.get_axis(0))
                
                if abs(stick_x) < 0.05:
                    target_l = master_speed
                    target_r = master_speed
                    blend_status = "BOTH"
                elif stick_x > 0:
                    target_l = master_speed
                    target_r = master_speed * (1 - stick_x)
                    blend_status = f"R {stick_x*100:.0f}%"
                else:
                    target_r = master_speed
                    target_l = master_speed * (1 + stick_x)
                    blend_status = f"L {-stick_x*100:.0f}%"
                
                current_l_speed = smooth_speed_change(current_l_speed, target_l, ACCELERATION_RATE, DECELERATION_RATE)
                current_r_speed = smooth_speed_change(current_r_speed, target_r, ACCELERATION_RATE, DECELERATION_RATE)
                
                motors.set_motors_percent(current_l_speed, current_r_speed)
                
                if master_speed > 0.5:
                    print(f" MANUAL | Speed: {master_speed:.1f}% | Blend: {blend_status} | L:{current_l_speed:.0f}% R:{current_r_speed:.0f}%", end='\r')
                
                # Draw center line
                cv2.line(frame, (int(fw/2), 0), (int(fw/2), fh), (0, 255, 255), 2)
                cv2.putText(frame, "MANUAL - Triangle=AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if SHOW_CAMERA:
                    cv2.imshow("Catamaran", frame)
            
            if SHOW_CAMERA and cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            
            clock.tick(40)
    
    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        motors.stop()
        if SHOW_CAMERA:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

# ====== MAIN ======
def main():
    
    # Setup hardware
    setup_pca()
    joystick = init_ps4() # Joystick = None if not found
    
    # Require joystick for 3-mode setup
    if not joystick:
        print(" PS4 controller required for all modes")
        print("  Aborting startup\n")
        return
    
    setup_camera()
    setup_model()
        
    # Create motor & servo controller
    motors = MotorController()
    motors.arm_motors()
    servo = ServoController()
    servo.close_net()

    # Menu
    while True:
        print("\n" + "=" * 70)
        print("SELECT MODE:")
        print("=" * 70)

        print(f"  1. PS4 Advanced (R2 speed + stick differential)")
        print(f"  2. AI Simple + R2 Speed + Stick Differential")
        print(f"  3. AI Smart Search + R2 Speed + Stick Differential")
        print(f"  4. Exit")
        
        choice = input("\nChoice (1-4): ").strip() # strips whitespace and returns characters only
        
        if choice == "1":
            if not joystick:
                print(" Need PS4 controller")
                continue
            mode_ps4_advanced(motors,joystick)
        
        elif choice == "2":
            if not joystick:
                print(" Need PS4 controller")
                continue
            mode_ai_simple_advanced(motors,servo,joystick)
        
        elif choice == "3":
            if not joystick:
                print(" Need PS4 controller")
                continue
            mode_ai_search_advanced(motors,servo,joystick)
        
        elif choice == "4":
            print("\n Ciao!")
            break
        
        else:
            print(" Invalid choice")
    
    # Cleanup
    print("\n Cleanup...")
    motors.cleanup()
    servo.cleanup() 

    if cam:
        if cam.isOpened():
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Flush events
            cam.release()

    if joystick:
       pygame.quit()

    print(" Done!")

#  Program call:
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program interrupted")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()