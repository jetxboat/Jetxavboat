"""
PS4 Controller DC 
Differential drive 
"""
import RPi.GPIO as GPIO # General Purpose Input Output Pins use.
import pygame # Reads PS4 joystick controller inputs.
import time # Handles timing and delays in the control loop.

# ====== GPIO PIN CONFIGURATION ====== 
ENA, ENB = 15, 33 # PWM pins  
IN1, IN2 = 36, 38  # Right motor
IN3, IN4 = 40, 37  # Left motor
PWM_FREQ = 100      # Hz

# ====== CONTROL PARAMETERS ======
MAX_SPEED = 100         # Maximum duty cycle %
DEADZONE = 0.15         # Joystick deadzone 
TURN_SENSITIVITY = 0.7  

# Global PWM objects
rpwm = None
lpwm = None

# ====== GPIO SETUP ======
def setup_pins():
    """Initialize GPIO pins and PWM"""
    global rpwm, lpwm
    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    
    pins = (ENA, ENB, IN1, IN2, IN3, IN4)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
    
    # Create PWM objects
    rpwm = GPIO.PWM(ENA, PWM_FREQ)
    lpwm = GPIO.PWM(ENB, PWM_FREQ)
    rpwm.start(0)
    lpwm.start(0)
    print("GPIO initialized")

# ====== MOTOR CONTROL FUNCTIONS ======
def set_motor_speed(right_speed, left_speed):
    """
    Set motor speeds with direction, Speed range: -100 to +100
    """
    # Clamp speeds
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
    left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    
    # Right motor
    if right_speed >= 0: # Positive 
        # Forward
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        rpwm.ChangeDutyCycle(abs(right_speed))
    else: # Negative
        # Backward
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        rpwm.ChangeDutyCycle(abs(right_speed))
    
    # Left motor
    if left_speed >= 0:
        # Forward
        GPIO.output(IN4, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        lpwm.ChangeDutyCycle(abs(left_speed))
    else:
        # Backward
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
    # Scale remaining range to full 0-1
    sign = 1 if value > 0 else -1
    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return sign * scaled

# ====== PS4 CONTROLLER SETUP ======
def init_ps4_controller():
    """Initialize PS4 controller"""
    pygame.init()
    pygame.joystick.init()
    
    joystick_count = pygame.joystick.get_count()
    
    if joystick_count == 0:
        print(" No controller found!")
        return None
    
    print(f"Found {joystick_count} controller(s)")
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    name = joystick.get_name()
    axes = joystick.get_numaxes()
    buttons = joystick.get_numbuttons()
    
    print(f"   Name: {name}")
    print(f"   Axes: {axes}")       
    print(f"   Buttons: {buttons}")
    
    return joystick

# ====== CONTROL MODES ======
def tank_drive(joystick):
    """
    Tank drive mode: Each stick controls one side
    Left stick Y-axis → Left motor
    Right stick Y-axis → Right motor
    """
    # PS4 Controller axes:
    # Axis 1: Left stick Y (up/down)
    # Axis 4: Right stick Y (up/down)
    
    left_axis = -joystick.get_axis(1)   # Invert (up = positive)
    right_axis = -joystick.get_axis(4)  # Invert (up = positive)
    
    # Apply deadzone
    left_axis = apply_deadzone(left_axis)
    right_axis = apply_deadzone(right_axis)
    
    # Convert to motor speeds
    left_speed = left_axis * MAX_SPEED
    right_speed = right_axis * MAX_SPEED
    
    return right_speed, left_speed

def arcade_drive(joystick):
    """
    Arcade drive mode: One stick controls everything
    Left stick Y-axis → Forward/Backward
    Left stick X-axis → Left/Right turning
    (More intuitive for most people!)
    """
    # PS4 Controller axes:
    # Axis 0: Left stick X (left/right)
    # Axis 1: Left stick Y (up/down)
    
    throttle = -joystick.get_axis(1)  # Forward/backward (invert Y)
    turn = joystick.get_axis(0)       # Left/right
    
    # Apply deadzone
    throttle = apply_deadzone(throttle)
    turn = apply_deadzone(turn)
    
    # Mix throttle and turn for differential steering, for smooth turning
    left_speed = throttle + (turn * TURN_SENSITIVITY)
    right_speed = throttle - (turn * TURN_SENSITIVITY)
    
    # Normalize if any value exceeds 1.0
    max_magnitude = max(abs(left_speed), abs(right_speed))
    if max_magnitude > 1.0:
        left_speed /= max_magnitude
        right_speed /= max_magnitude
    
    # Convert to motor speeds
    left_speed *= MAX_SPEED
    right_speed *= MAX_SPEED
    
    return right_speed, left_speed

# ====== MAIN PROGRAM ======
def main():
    print("=" * 60)
    print("PS4 CONTROLLER CONTROL")
    print("=" * 60)
    
    # Setup
    setup_pins()
    joystick = init_ps4_controller()
    
    if joystick is None:
        GPIO.cleanup()
        return
    
    # Select control mode
    print("\n SELECT CONTROL MODE:")
    print("   1. Arcade Drive (Left stick: forward/turn)")
    print("   2. Tank Drive (Each stick: one motor)")
    
    mode_choice = input("\nEnter choice (1 or 2): ").strip() # Strip whitespace returns characters
    
    if mode_choice == "1":
        drive_mode = arcade_drive
        print("\n Arcade Drive Mode")
        print("   Left Stick Up/Down: Forward/Backward")
        print("   Left Stick Left/Right: Turn")
    else:
        drive_mode = tank_drive
        print("\n Tank Drive Mode")
        print("   Left Stick: Left Motor")
        print("   Right Stick: Right Motor")
    
    print("\n CONTROLS:")
    print("   Circle: Emergency Stop")
    print("   Options Button: Quit")
    print("\n Ready! Move sticks to Drive")
    print("=" * 60 + "\n")
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get motor speeds from selected drive mode
            right_speed, left_speed = drive_mode(joystick)
            
            # Check buttons
            # Button 1: Circle (○) - Emergency stop
            if joystick.get_button(1):
                print(" EMERGENCY STOP")
                stop_motors()
                time.sleep(0.1)
                continue
            
            # Button 9: Options - Quit
            if joystick.get_button(9):
                print(" Quitting...")
                running = False
                break
            
            # Set motor speeds
            set_motor_speed(right_speed, left_speed)
            
            # Display speeds (throttle display)
            if abs(right_speed) > 1 or abs(left_speed) > 1:
                print(f" R: {right_speed:+6.1f}% | L: {left_speed:+6.1f}%", end='\r')
            
            # Limit loop rate
            clock.tick(50)  
    
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    
    finally:
        # Cleanup
        print("\n Cleaning up...")
        stop_motors()
        rpwm.stop()
        lpwm.stop()
        GPIO.cleanup()
        pygame.quit()
        print(" Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()