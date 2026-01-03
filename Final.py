Jetxavboat
"""
CATAMARAN BI BLDC BOAT CONTROLLER
5 modes:
0. Manual Pulse Test (Bidirectional)
1. PS4 Advanced (R2=speed, left stick=differential control)
2. AI Simple + R2 FWD Speed + L2 REV Speed + Stick Differential
3. AI Smart Search + R2 FWD Speed  + L2 REV Speed+ Stick Differential
4. Timed Navigation (Toggle MANUAL ↔ AUTO)
"""

import time
import board
import pygame
import cv2
import numpy as np
import busio
from adafruit_pca9685 import PCA9685
from ultralytics import YOLO

# ====== MOTOR CONFIGURATION ======
FREQUENCY = 50

MOTOR_L_CHANNEL = 0 
MOTOR_R_CHANNEL = 1
SERVO1_CHANNEL = 3

# DISCOVERED VALUES 

# BI MOTORS
MOTOR_NEUTRAL = 4700      # Dead stop
MOTOR_FWD_MAX = 5500      # Full forward
MOTOR_REV_MAX = 3900      # Full reverse

# SERVO CONFIGURATION
SERVO1_MIN = 1800
SERVO1_CENTER = 4000
SERVO1_MAX = 8400

# SAFETY LIMIT
MOTOR_SAFE_FWD_MAX = 4900
MOTOR_SAFE_REV_MAX = 3900

# START LIMIT
INITIAL_MAX_SPEED = 65 # % Changed with each test

# ====== NAVIGATION SPEEDS ======
FORWARD_SPEED = 40  # %
TURN_SPEED = 30     # % BI

# Motor trim
MOTOR_L_TRIM = 1.0
MOTOR_R_TRIM = 1.0

# ====== PS4 SETTINGS ======
DEADZONE = 0.15
R2_SENSITIVITY = 1.3

# ====== AI PARAMETERS ======
MODEL = "best7.pt"
AI_APPROACH_SPEED = 30          
AI_CLOSE_APPROACH_SPEED = 20    
AI_CENTERING_THRESHOLD = 0.25
AI_CLOSE_DISTANCE = 0.4
AI_CONFIDENCE_THRESHOLD = 0.8
BOTTLE_CLASS_ID = 0

# ====== PID PARAMETERS ======
KP = 0.5 # 0.3
KI = 0.007 # 0.005
KD = 0.5 # 0.15

# ====== SEARCH PARAMETERS ======
LOOK_TURN_SPEED = 30 # BI
LOOK_TURN_FRAMES_45 = 25
LOOK_WAIT_FRAMES = 40
SEARCH_SPIN_SPEED = 40 # 360 

# ====== SMOOTHING ======
ACCELERATION_RATE = 2.3
DECELERATION_RATE = 4.0

# Smooth speed tracking
current_l_speed = 0.0
current_r_speed = 0.0

# ====== DISPLAY ======
SHOW_CAMERA = True  # False for SSH

print("=" * 50)
print(" CATAMARAN BLDC BOAT CONTROLLER ")
print("=" * 50)

# Globals
pca = None
cam = None
model = None
ai_enabled = False
pid_integral = 0
pid_last_error = 0
frames_without_target = 0
nav_mode = "MANUAL"  

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
    return None

# ====== BI MOTOR CONTROLLER ======
class MotorController:
    def __init__(self):
        self.l_ch = pca.channels[MOTOR_L_CHANNEL]
        self.r_ch = pca.channels[MOTOR_R_CHANNEL]
        
        # Calculate safe percentage range
        fwd_range = MOTOR_SAFE_FWD_MAX - MOTOR_NEUTRAL
        total_fwd_range = MOTOR_FWD_MAX - MOTOR_NEUTRAL
        self.safe_fwd_pct = (fwd_range / total_fwd_range) * 100
        
        rev_range = MOTOR_NEUTRAL - MOTOR_SAFE_REV_MAX
        total_rev_range = MOTOR_NEUTRAL - MOTOR_REV_MAX
        self.safe_rev_pct = (rev_range / total_rev_range) * 100
        
        print(f"Motor Limits (Bidirectional):")
        print(f"  Reverse: {MOTOR_SAFE_REV_MAX}-{MOTOR_REV_MAX}µs (0-{self.safe_rev_pct:.0f}%)")
        print(f"  Neutral: {MOTOR_NEUTRAL}µs (0%)")
        print(f"  Forward: {MOTOR_NEUTRAL}-{MOTOR_SAFE_FWD_MAX}µs (0-{self.safe_fwd_pct:.0f}%)")
        print(f"  Initial speed limit: ±{INITIAL_MAX_SPEED}%\n")
    
    def percent_to_pulse(self, pct_l, pct_r):
        # Convert -100 to +100 percent to PWM pulse
        # Apply motor trim
        pct_l *= MOTOR_L_TRIM
        pct_r *= MOTOR_R_TRIM
        
        # Clamp to safe limits
        pct_l = max(-self.safe_rev_pct, min(self.safe_fwd_pct, pct_l))
        pct_r = max(-self.safe_rev_pct, min(self.safe_fwd_pct, pct_r))
        
        # Map percentage to pulse - LEFT MOTOR
        if pct_l >= 0:  # Forward
            pulse_l = MOTOR_NEUTRAL + (MOTOR_FWD_MAX - MOTOR_NEUTRAL) * pct_l / 100
        else:  # Reverse
            pulse_l = MOTOR_NEUTRAL + (MOTOR_NEUTRAL - MOTOR_REV_MAX) * pct_l / 100
        
        # Map percentage to pulse - RIGHT MOTOR
        if pct_r >= 0:  # Forward
            pulse_r = MOTOR_NEUTRAL + (MOTOR_FWD_MAX - MOTOR_NEUTRAL) * pct_r / 100
        else:  # Reverse
            pulse_r = MOTOR_NEUTRAL + (MOTOR_NEUTRAL - MOTOR_REV_MAX) * pct_r / 100
        
        # Safety clamp to absolute limits
        pulse_l = max(MOTOR_REV_MAX, min(MOTOR_FWD_MAX, int(pulse_l)))
        pulse_r = max(MOTOR_REV_MAX, min(MOTOR_FWD_MAX, int(pulse_r)))
        
        return pulse_l, pulse_r
    
    def set_motors_percent(self, pct_l, pct_r):
        # Set motors using -100 to +100 percent
        pulse_l, pulse_r = self.percent_to_pulse(pct_l, pct_r)
        self.l_ch.duty_cycle = pulse_l
        self.r_ch.duty_cycle = pulse_r
        return pulse_l, pulse_r
    
    def set_motors_pulse(self, pulse_l, pulse_r):
        # Set motors using raw PWM pulses
        pulse_l = max(MOTOR_REV_MAX, min(MOTOR_FWD_MAX, int(pulse_l)))
        pulse_r = max(MOTOR_REV_MAX, min(MOTOR_FWD_MAX, int(pulse_r)))
        self.l_ch.duty_cycle = pulse_l
        self.r_ch.duty_cycle = pulse_r
        return pulse_l, pulse_r
    
    def arm_motors(self):
        # Arm ESCs for bidirectional mode
        print("\n Arming Bidirectional ESCs...")
        print("   Setting NEUTRAL throttle...")
        
        # Set to neutral (not minimum)
        self.l_ch.duty_cycle = MOTOR_NEUTRAL
        self.r_ch.duty_cycle = MOTOR_NEUTRAL
        
        print(f"   Left:  {MOTOR_NEUTRAL}µs (neutral)")
        print(f"   Right: {MOTOR_NEUTRAL}µs (neutral)")
        
        print("   Waiting for ESC initialization...")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("   ✓ Motors armed !\n")
    
    def stop(self):
        # Stop motors (return to neutral)
        self.l_ch.duty_cycle = MOTOR_NEUTRAL
        self.r_ch.duty_cycle = MOTOR_NEUTRAL
    
    def cleanup(self):
        # Safe shutdown
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
        self.set_servo1_percent(50)
    
    def close_net(self):
        self.set_servo1_percent(0)
    
    def stop(self):
        self.servo1.duty_cycle = 0
    
    def cleanup(self):
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

def smooth_speed_change(current, target, accel_rate=ACCELERATION_RATE, decel_rate=DECELERATION_RATE):
    diff = target - current
    
    if abs(diff) < 0.5:
        return target
    
    if diff > 0:
        return current + min(accel_rate, diff)
    else:
        return current + max(-decel_rate, diff)

def smooth_turn(base_speed, turn_amt):
    if turn_amt > 0:
        l_speed = base_speed
        r_speed = base_speed * (1 - abs(turn_amt) * 0.7)  
    else:
        l_speed = base_speed * (1 - abs(turn_amt) * 0.7)
        r_speed = base_speed
    
    return l_speed, r_speed

# ====== NAVIGATION OVERLAY ======
def draw_camera_overlay(frame, mode, action="", speed_l=0, speed_r=0, time_info=""):
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Mode indicator
    mode_color = (0, 255, 0) if mode == "AUTO" else (255, 255, 0)
    cv2.putText(frame, f"MODE: {mode}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    # Action info
    if action:
        cv2.putText(frame, action, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Motor speeds
    speed_text = f"L: {speed_l:.0f}%  R: {speed_r:.0f}%"
    cv2.putText(frame, speed_text, (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Time info
    if time_info:
        cv2.putText(frame, time_info, (w - 200, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Direction indicator
    center_x, center_y = w // 2, h // 2
    if mode == "AUTO":
        if "FORWARD" in action:
            cv2.arrowedLine(frame, (center_x, center_y + 30), (center_x, center_y - 30), 
                           (0, 255, 0), 3, tipLength=0.3)
        elif "RIGHT" in action:
            cv2.arrowedLine(frame, (center_x - 30, center_y), (center_x + 30, center_y), 
                           (0, 255, 255), 3, tipLength=0.3)
        elif "LEFT" in action:
            cv2.arrowedLine(frame, (center_x + 30, center_y), (center_x - 30, center_y), 
                           (255, 0, 255), 3, tipLength=0.3)
    
    return frame

# ====== BOTTLE DETECTION ======
def detect_bottle(results, fw, fh):
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            
            if conf > AI_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_cen = (x1 + x2) / 2
                y_cen = (y1 + y2) / 2
                
                bbox_area = (x2 - x1) * (y2 - y1)
                area_ratio = bbox_area / (fw * fh)
                
                return True, x_cen, y_cen, conf, area_ratio
    
    return False, None, None, None, None

# ====== PID STEERING ======
def pid_steer_to_bottle(bottle_x, fw, area_ratio):
    global pid_integral, pid_last_error
    
    fc = fw / 2
    error = (bottle_x - fc) / fc
    
   # # If very off-center, rotate in place first
   #  if abs(error) > 0.5:  # >50% off-center
   #      if error > 0:
   #         return LOOK_TURN_SPEED, -LOOK_TURN_SPEED, "⟳ CENTERING RIGHT"
   #     else:
   #         return -LOOK_TURN_SPEED, LOOK_TURN_SPEED, "⟲ CENTERING LEFT"
        
    pid_integral += error
    pid_integral = max(-1, min(1, pid_integral))
    
    deriv = error - pid_last_error
    pid_last_error = error
    
    correction = (KP * error) + (KI * pid_integral) + (KD * deriv)
    correction = max(-1, min(1, correction))
    
    # ===== PREDICTIVE SLOWDOWN =====
    # If error is small AND derivative shows we're converging fast, reduce speed
    if abs(error) < 0.15 and abs(deriv) > 0.02:
        # We're close to center AND moving fast toward it → SLOW DOWN
        base_speed_multiplier = 0.6  # Reduce to 60% speed
    else:
        base_speed_multiplier = 1.0
    
    # Apply distance-based speed
    if area_ratio > AI_CLOSE_DISTANCE:
        base = AI_CLOSE_APPROACH_SPEED * base_speed_multiplier
        status = " SLOW"
    else:
        base = AI_APPROACH_SPEED * base_speed_multiplier
        status = " APPROACH"

    # ===== DEAD ZONE FOR CENTERED =====
    if abs(error) < AI_CENTERING_THRESHOLD:
        l_spd = base
        r_spd = base
        status = " CENTERED - " + status
        # Reset integral when centered to prevent overshoot
        pid_integral = 0  
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
        # return LOOK_TURN_SPEED, 0, " TURN RIGHT 45°"
        
        # OPTION B: Counter-rotating
        return LOOK_TURN_SPEED, -LOOK_TURN_SPEED, "TURN RIGHT 45°"
    
    elif f < p3_end:
        return 0, 0, f" LOOK RIGHT ({f-p2_end}/{LOOK_WAIT_FRAMES})"
    
    elif f < p4_end:
        # OPTION A: One motor only
        # return 0, LOOK_TURN_SPEED, " RETURN CENTER"
        
        # OPTION B: Counter-rotating
        return -LOOK_TURN_SPEED, LOOK_TURN_SPEED, "RETURN CENTER"
    
    elif f < p5_end:
        return 0, 0, f" CENTER ({f-p4_end}/{LOOK_WAIT_FRAMES})"
    
    elif f < p6_end:
        # OPTION A: One motor only
        # return 0, LOOK_TURN_SPEED, " TURN LEFT 45°"
        
        # OPTION B: Counter-rotating
        return -LOOK_TURN_SPEED, LOOK_TURN_SPEED, "↺ TURN LEFT 45°"
    
    elif f < p7_end:
        return 0, 0, f" LOOK LEFT ({f-p6_end}/{LOOK_WAIT_FRAMES})"
    
    elif f < p8_end:
        # OPTION A: One motor only (current)
        # return LOOK_TURN_SPEED, 0, " RETURN CENTER"
        
        # OPTION B: Counter-rotating
        return LOOK_TURN_SPEED, -LOOK_TURN_SPEED, " RETURN CENTER"
    
    elif f < p9_end:
        return 0, 0, f" FINAL CHECK ({f-p8_end}/{LOOK_WAIT_FRAMES})"
    
    elif f < p10_end:
        # return SEARCH_SPIN_SPEED, 0, " SPIN 360°"
    
        # 360° search - counter-rotating
        return SEARCH_SPIN_SPEED, -SEARCH_SPIN_SPEED, " 360° SEARCH"
    
    else:
        frames_without_target = 0  # Reset to loop search
        return 0, 0, " SEARCH RESTART"

# ====== NAVIGATION MOVEMENTS ======
def move_forward(motors, speed, duration, start_time):
    global current_l_speed, current_r_speed
    
    elapsed = time.time() - start_time
    
    if elapsed < duration:
        current_l_speed = smooth_speed_change(current_l_speed, speed)
        current_r_speed = smooth_speed_change(current_r_speed, speed)
        
        motors.set_motors_percent(current_l_speed, current_r_speed)
        
        print(f" FORWARD | Speed: {speed}% | {elapsed:.1f}s / {duration}s", end='\r')
        return False, f"FORWARD {speed}%", f"{elapsed:.1f}s / {duration}s"
    else:
        return True, "", ""

def turn_right(motors, speed, duration, start_time):
    global current_l_speed, current_r_speed
    
    elapsed = time.time() - start_time
    
    if elapsed < duration:
        current_l_speed = smooth_speed_change(current_l_speed, speed)
        current_r_speed = smooth_speed_change(current_r_speed, -speed)
        
        motors.set_motors_percent(current_l_speed, current_r_speed)
        
        print(f"TURN RIGHT | {elapsed:.1f}s / {duration}s", end='\r')
        return False, "TURN RIGHT", f"{elapsed:.1f}s / {duration}s"
    else:
        return True, "", ""

def turn_left(motors, speed, duration, start_time):
    global current_l_speed, current_r_speed
    
    elapsed = time.time() - start_time
    
    if elapsed < duration:
        current_l_speed = smooth_speed_change(current_l_speed, -speed)
        current_r_speed = smooth_speed_change(current_r_speed, speed)
        
        motors.set_motors_percent(current_l_speed, current_r_speed)
        
        print(f"TURN LEFT | {elapsed:.1f}s / {duration}s", end='\r')
        return False, "TURN LEFT", f"{elapsed:.1f}s / {duration}s"
    else:
        return True, "", ""

# ====== NAVIGATION SEQUENCE ======
def execute_navigation_sequence(motors, joystick):
    global nav_mode, current_l_speed, current_r_speed
    
    sequence = [
        ("forward", FORWARD_SPEED, 6),
        ("right", TURN_SPEED, 5),
        ("forward", FORWARD_SPEED, 6),
        ("right", TURN_SPEED, 5),
        ("forward", FORWARD_SPEED, 6),
        ("right", TURN_SPEED, 5),
        ("forward", FORWARD_SPEED, 6),
        ("stop", 0, 0)
    ]
    
    step = 0
    start_time = time.time()
    action_text = ""
    time_text = ""

    print("\n Starting Navigation Sequence...\n")
    
    while step < len(sequence) and nav_mode == "AUTO":
        # Read camera frame
        if cam and cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                frame = None
        else:
            frame = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        
        if joystick:
            # Triangle - Back to manual
            if joystick.get_button(2):
                nav_mode = "MANUAL"
                motors.stop()
                current_l_speed = 0
                current_r_speed = 0
                print("\n\n MANUAL MODE - Navigation aborted")
                time.sleep(0.3)
                return
            
            # Circle - Emergency stop
            if joystick.get_button(1):
                motors.stop()
                current_l_speed = 0
                current_r_speed = 0
                print("\n\n EMERGENCY STOP")
                time.sleep(0.5)
                continue

            # Options - Quit
            if joystick.get_button(9):
                break
        
        # Execute current step
        action, speed, duration = sequence[step]
        
        if action == "forward":
            finished, action_text, time_text = move_forward(motors, speed, duration, start_time)
        elif action == "right":
            finished, action_text, time_text = turn_right(motors, speed, duration, start_time)
        elif action == "left":
            finished, action_text, time_text = turn_left(motors, speed, duration, start_time)
        elif action == "stop":
            motors.stop()
            current_l_speed = 0
            current_r_speed = 0
            action_text = "SEQUENCE COMPLETE"
            print("\n\n Navigation Sequence Complete!")
            
            # Show completion on camera for 2 seconds
            if frame is not None and SHOW_CAMERA:
                for _ in range(40):
                    if cam and cam.isOpened():
                        ret, frame = cam.read()
                        if ret:
                            frame = draw_camera_overlay(frame, "AUTO", action_text, 0, 0, "")
                            cv2.imshow("Catamaran Navigation", frame)
                            cv2.waitKey(50)
            else:
                time.sleep(2)
            break
        
        # Display camera with overlay
        if frame is not None and SHOW_CAMERA:
            frame = draw_camera_overlay(frame, "AUTO", action_text, 
                                       current_l_speed, current_r_speed, time_text)
            cv2.imshow("Catamaran Navigation", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                nav_mode = "MANUAL"
                return
        
        if finished:
            print()
            step += 1
            start_time = time.time()
        
        time.sleep(0.05)

# ====== CONTROL MODES ======

def mode_manual_pulse(motors):
    # Manual pulse testing for bidirectional thrusters
    print("\n" + "=" * 50)
    print("MODE: MANUAL PULSE TEST (BIDIRECTIONAL)")
    print("=" * 50)
    print(f"\nPulse ranges:")
    print(f"  Reverse: {MOTOR_REV_MAX}µs")
    print(f"  Neutral: {MOTOR_NEUTRAL}µs")
    print(f"  Forward: {MOTOR_NEUTRAL} - {MOTOR_FWD_MAX}µs")
    print("Commands:")
    print("  #### ####  : Set L and R pulse (e.g., 4700 4700)")
    print("  test       : Run calibration sequence")
    print("  q          : Quit\n")
    
    try:
        while True:
            cmd = input(f"Pulse [L R] or 'test': ").strip().lower()
            
            if cmd == 'q':
                break
            
            elif cmd == 'test':
                print("\n Running calibration sequence...")
                print("   Testing NEUTRAL (should not spin)...")
                motors.set_motors_pulse(MOTOR_NEUTRAL, MOTOR_NEUTRAL)
                input("   Press ENTER to continue...")
                
                print("   Testing FORWARD 30%...")
                fwd_30 = int(MOTOR_NEUTRAL + (MOTOR_FWD_MAX - MOTOR_NEUTRAL) * 0.3)
                motors.set_motors_pulse(fwd_30, fwd_30)
                input("   Press ENTER to continue...")
                
                print("   Testing REVERSE 30%...")
                rev_30 = int(MOTOR_NEUTRAL - (MOTOR_NEUTRAL - MOTOR_REV_MAX) * 0.3)
                motors.set_motors_pulse(rev_30, rev_30)
                input("   Press ENTER to stop...")
                
                motors.stop()
                print("   ✓ Test complete\n")
            
            else:
                try:
                    parts = cmd.split()
                    if len(parts) == 2:
                        pl = int(parts[0])
                        pr = int(parts[1])
                        
                        # Safety check
                        if (pl < MOTOR_REV_MAX or pl > MOTOR_FWD_MAX or
                            pr < MOTOR_REV_MAX or pr > MOTOR_FWD_MAX):
                            print(f"  ✗ Out of range! Use {MOTOR_REV_MAX}-{MOTOR_FWD_MAX}")
                            continue
                        
                        al, ar = motors.set_motors_pulse(pl, pr)
                        
                        # Calculate percentages for display
                        if pl >= MOTOR_NEUTRAL:
                            l_pct = ((pl - MOTOR_NEUTRAL) / (MOTOR_FWD_MAX - MOTOR_NEUTRAL)) * 100
                        else:
                            l_pct = -((MOTOR_NEUTRAL - pl) / (MOTOR_NEUTRAL - MOTOR_REV_MAX)) * 100
                        
                        if pr >= MOTOR_NEUTRAL:
                            r_pct = ((pr - MOTOR_NEUTRAL) / (MOTOR_FWD_MAX - MOTOR_NEUTRAL)) * 100
                        else:
                            r_pct = -((MOTOR_NEUTRAL - pr) / (MOTOR_NEUTRAL - MOTOR_REV_MAX)) * 100
                        
                        print(f"  ✓ Set: L={al}µs ({l_pct:+.0f}%), R={ar}µs ({r_pct:+.0f}%)")
                    
                    else:
                        print("  ✗ Invalid - use: 4700 4700")
                
                except ValueError:
                    print("  ✗ Invalid - use numbers: 4700 4700")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
    
    except KeyboardInterrupt:
        print("\n Interrupted")
    finally:
        motors.stop()

def mode_ps4_advanced(motors, joy):
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

                 # Controller disconnection detection
                if event.type == pygame.JOYDEVICEREMOVED:
                   print("\n\n CONTROLLER DISCONNECTED - EMERGENCY STOP")
                   motors.stop()
                   current_l_speed = 0
                   current_r_speed = 0
                   running = False
                   break
            
            # Check L2 (reverse) 
            l2_raw = (joy.get_axis(2) + 1) / 2
            if l2_raw > 0.1:  # L2 pressed = reverse
               master_speed = -l2_raw * INITIAL_MAX_SPEED
            else:  # check R2 (forward)
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
            
            current_l_speed = smooth_speed_change(current_l_speed, target_l)
            current_r_speed = smooth_speed_change(current_r_speed, target_r)
            
            motors.set_motors_percent(current_l_speed, current_r_speed)
            
            if joy.get_button(1):
                current_l_speed = 0
                current_r_speed = 0
                motors.stop()
                print("\n STOP")
                time.sleep(0.3)
            
            if joy.get_button(9):
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

                # Controller disconnection detection
                if event.type == pygame.JOYDEVICEREMOVED:
                   print("\n\n CONTROLLER DISCONNECTED - EMERGENCY STOP")
                   motors.stop()
                   current_l_speed = 0
                   current_r_speed = 0
                   running = False
                   break

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
            
            if joy.get_button(1):
                motors.stop()
                current_l_speed = 0
                current_r_speed = 0
                print("\n EMERGENCY STOP")
                time.sleep(0.3)
                continue
            
            if joy.get_button(9):
                running = False
                break
            
            if ai_enabled:
                results = model(frame, conf=AI_CONFIDENCE_THRESHOLD, verbose=False)
                annotated = results[0].plot()
                
                detected, bx, by, conf, area_ratio = detect_bottle(results, fw, fh)
 
                if detected:
                    # BOTTLE DETECTED 
                    frames_without_target = 0
                    servo.open_net()
                    
                    target_l, target_r, status = pid_steer_to_bottle(bx, fw, area_ratio)
                    
                    # Smooth speed changes
                    current_l_speed = smooth_speed_change(current_l_speed, target_l, ACCELERATION_RATE, DECELERATION_RATE)
                    current_r_speed = smooth_speed_change(current_r_speed, target_r, ACCELERATION_RATE, DECELERATION_RATE)
                    
                    motors.set_motors_percent(current_l_speed, current_r_speed)
                    
                    # Draw visuals 
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
                    # NO BOTTLE 
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
                # MANUAL MODE - L2/R2 Speed + Stick Differential
                l2_raw = (joy.get_axis(2) + 1) / 2
                if l2_raw > 0.1:
                    master_speed = -l2_raw * INITIAL_MAX_SPEED
                else:
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

              # Controller disconnection detection
                if event.type == pygame.JOYDEVICEREMOVED:
                   print("\n\n CONTROLLER DISCONNECTED - EMERGENCY STOP")
                   motors.stop()
                   current_l_speed = 0
                   current_r_speed = 0
                   running = False
                   break

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
                
                detected, bx, by, conf, area_ratio = detect_bottle(results, fw, fh)
                
                if detected:
                    # TARGET FOUND - Draw visuals
                    frames_without_target = 0
                    servo.open_net()
            
                    target_l, target_r, status = pid_steer_to_bottle(bx, fw, area_ratio)
                    
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
                    # TARGET LOST - SEARCH PATTERN 
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
                 # MANUAL MODE - L2/R2 Speed + Stick Differential
                l2_raw = (joy.get_axis(2) + 1) / 2
                if l2_raw > 0.1:
                    master_speed = -l2_raw * INITIAL_MAX_SPEED
                else:
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

def mode_navigation(motors, joystick):
    # Mode 4: Timed Navigation with Manual/Auto Toggle
    global nav_mode, current_l_speed, current_r_speed
    
    print("\n" + "=" * 70)
    print("MODE 4: TIMED NAVIGATION")
    print("=" * 70)
    print("\n Triangle: Toggle MANUAL ↔ AUTO Navigation")
    print("   Circle: Emergency Stop")
    print("   R2 + Left Stick: Manual control")
    print("   Options: Quit\n")
    print(" Starting in MANUAL mode\n")
    
    nav_mode = "MANUAL"
    current_l_speed = 0
    current_r_speed = 0
    
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            # Read camera frame
            if cam and cam.isOpened():
                ret, frame = cam.read()
                if not ret:
                    frame = None
            else:
                frame = None
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            # Controller disconnection detection
                if event.type == pygame.JOYDEVICEREMOVED:   
                   print("\n\n CONTROLLER DISCONNECTED - EMERGENCY STOP")
                   motors.stop()
                   current_l_speed = 0
                   current_r_speed = 0
                   running = False
                   break

            if joystick:
                # Triangle - Toggle mode
                if joystick.get_button(2):
                    if nav_mode == "MANUAL":
                        nav_mode = "AUTO"
                        current_l_speed = 0
                        current_r_speed = 0
                        print("\n\n AUTO NAVIGATION MODE\n")
                        time.sleep(0.5)
                        execute_navigation_sequence(motors, joystick)
                        # After sequence completes, return to manual
                        nav_mode = "MANUAL"
                        print("\n\n MANUAL MODE\n")
                    time.sleep(0.3)
                
                # Circle - Emergency stop
                if joystick.get_button(1):
                    motors.stop()
                    current_l_speed = 0
                    current_r_speed = 0
                    print("\n EMERGENCY STOP")
                    time.sleep(0.3)
                    continue
                
                # Options - Quit back to menu
                if joystick.get_button(9):
                    running = False
                    break
            
            # Execute current mode
            if nav_mode == "MANUAL" and joystick:

                # MANUAL MODE - L2/R2 Speed + Stick Differential
                l2_raw = (joystick.get_axis(2) + 1) / 2
                if l2_raw > 0.1:
                    master_speed = -l2_raw * INITIAL_MAX_SPEED
                else:
                    r2_raw = (joystick.get_axis(5) + 1) / 2
                    r2_raw = min(1.0, r2_raw * R2_SENSITIVITY)
                    master_speed = r2_raw * INITIAL_MAX_SPEED
                
                stick_x = apply_deadzone(joystick.get_axis(0))
                
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
                
                current_l_speed = smooth_speed_change(current_l_speed, target_l)
                current_r_speed = smooth_speed_change(current_r_speed, target_r)
                
                motors.set_motors_percent(current_l_speed, current_r_speed)
                
                if master_speed > 0.5:
                    print(f" MANUAL | Speed: {master_speed:.1f}% | {blend_status} | L:{current_l_speed:.0f}% R:{current_r_speed:.0f}%", end='\r')
                
                # Display camera in manual mode
                if frame is not None and SHOW_CAMERA:
                    frame = draw_camera_overlay(frame, "MANUAL", "R2 + Stick Control", 
                                               current_l_speed, current_r_speed, "")
                    cv2.imshow("Catamaran Navigation", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        running = False
            
            clock.tick(20)
    
    except KeyboardInterrupt:
        print("\n\n Interrupted")
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

        print(f"  0. Manual Pulse Test (Bidirectional)")
        print(f"  1. PS4 Advanced (R2 speed + stick differential)")
        print(f"  2. AI Simple Chase")
        print(f"  3. AI Smart Search")
        print(f"  4. Navigation Sequence")
        print(f"  5. Exit")
        
        choice = input("\nChoice (0-5): ").strip() # strips whitespace and returns characters only
        
        if choice == "0":
            mode_manual_pulse(motors)

        elif choice == "1":
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
            if not joystick:
                print(" Need PS4 controller")
                continue
            mode_navigation(motors, joystick)
        
        elif choice == "5":
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