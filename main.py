import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
from pygame.locals import *
import math

class Hand3DController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 3D object parameters
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.zoom = -7
        self.position_x = 0
        self.position_y = 0
        
        # Gesture tracking
        self.prev_hand_distance = None
        self.prev_rotation_point = None
        self.current_shape = 0  # 0: cube, 1: sphere, 2: torus, 3: pyramid
        
        # Initialize Pygame and OpenGL
        pygame.init()
        self.display = (1280, 720)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Futuristic 3D Hand Controller")
        
        # OpenGL setup
        self.setup_opengl()
        
        # Colors for futuristic look
        self.neon_colors = [
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 0.0),  # Green
            (1.0, 1.0, 0.0),  # Yellow
        ]
        self.current_color = 0
        
    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLight(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
        glLight(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        # Enable line smoothing for futuristic wireframe
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(2.0)
        
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_hand_closed(self, hand_landmarks):
        """Check if hand is closed (fist or two fingers closed)"""
        # Get fingertips and base points
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_bases = [2, 5, 9, 13, 17]
        
        closed_count = 0
        for tip, base in zip(fingertips[1:], finger_bases[1:]):  # Skip thumb
            tip_y = hand_landmarks.landmark[tip].y
            base_y = hand_landmarks.landmark[base].y
            if tip_y > base_y:  # Finger is bent
                closed_count += 1
                
        return closed_count >= 2  # At least 2 fingers closed (fist or two fingers)
    
    def get_hand_center(self, hand_landmarks):
        """Get center point of hand"""
        x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        y = sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        return (x, y)
    
    def process_gestures(self, results):
        """Process hand gestures and update 3D object"""
        if not results.multi_hand_landmarks:
            self.prev_hand_distance = None
            self.prev_rotation_point = None
            return
        
        hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            center = self.get_hand_center(hand_landmarks)
            is_closed = self.is_hand_closed(hand_landmarks)
            hands_data.append({
                'landmarks': hand_landmarks,
                'center': center,
                'closed': is_closed
            })
        
        # Two hands detected - Zoom control
        if len(hands_data) == 2:
            hand1_center = hands_data[0]['center']
            hand2_center = hands_data[1]['center']
            
            current_distance = self.calculate_distance(hand1_center, hand2_center)
            
            if self.prev_hand_distance is not None:
                distance_change = current_distance - self.prev_hand_distance
                self.zoom += distance_change * 15  # Increased from 3 to 15 for faster zoom
                self.zoom = max(-15, min(-2, self.zoom))  # Limit zoom range
            
            self.prev_hand_distance = current_distance
            
            # Rotation with both hands open
            if not hands_data[0]['closed'] and not hands_data[1]['closed']:
                mid_x = (hand1_center[0] + hand2_center[0]) / 2
                mid_y = (hand1_center[1] + hand2_center[1]) / 2
                
                if self.prev_rotation_point is not None:
                    delta_x = (mid_x - self.prev_rotation_point[0]) * 360
                    delta_y = (mid_y - self.prev_rotation_point[1]) * 360
                    
                    self.rotation_y += delta_x
                    self.rotation_x += delta_y
                
                self.prev_rotation_point = (mid_x, mid_y)
        
        # One hand detected - Movement or rotation
        elif len(hands_data) == 1:
            hand = hands_data[0]
            
            # If hand is closed (fist or two fingers) - move object smoothly
            if hand['closed']:
                if self.prev_rotation_point is not None:
                    delta_x = (hand['center'][0] - self.prev_rotation_point[0]) * 5  # Reduced from 10 to 5 for smoother movement
                    delta_y = (hand['center'][1] - self.prev_rotation_point[1]) * 5  # Reduced from 10 to 5 for smoother movement
                    
                    self.position_x += delta_x
                    self.position_y -= delta_y
                
                self.prev_rotation_point = hand['center']
            
            # If hand is open - rotate
            else:
                if self.prev_rotation_point is not None:
                    delta_x = (hand['center'][0] - self.prev_rotation_point[0]) * 360
                    delta_y = (hand['center'][1] - self.prev_rotation_point[1]) * 360
                    
                    self.rotation_y += delta_x
                    self.rotation_x += delta_y
                
                self.prev_rotation_point = hand['center']
            
            self.prev_hand_distance = None
        else:
            self.prev_hand_distance = None
            self.prev_rotation_point = None
    
    def draw_cube_wireframe(self):
        """Draw a futuristic wireframe cube"""
        vertices = [
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
        ]
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
        
    def draw_sphere_wireframe(self, radius=1, slices=20, stacks=20):
        """Draw a futuristic wireframe sphere"""
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = radius * math.sin(lat0)
            zr0 = radius * math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = radius * math.sin(lat1)
            zr1 = radius * math.cos(lat1)
            
            glBegin(GL_LINE_LOOP)
            for j in range(slices):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                
                glVertex3f(x * zr0, y * zr0, z0)
            glEnd()
            
    def draw_torus_wireframe(self, inner=0.5, outer=1.0, sides=20, rings=20):
        """Draw a futuristic wireframe torus"""
        for i in range(rings):
            theta = 2.0 * math.pi * i / rings
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            
            glBegin(GL_LINE_LOOP)
            for j in range(sides):
                phi = 2.0 * math.pi * j / sides
                cos_phi = math.cos(phi)
                sin_phi = math.sin(phi)
                
                r = outer + inner * cos_phi
                x = r * cos_theta
                y = r * sin_theta
                z = inner * sin_phi
                
                glVertex3f(x, y, z)
            glEnd()
    
    def draw_pyramid_wireframe(self):
        """Draw a futuristic wireframe pyramid"""
        vertices = [
            [0, 1.5, 0],    # Top
            [1, -1, 1],     # Base
            [1, -1, -1],
            [-1, -1, -1],
            [-1, -1, 1]
        ]
        
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # From top to base
            (1, 2), (2, 3), (3, 4), (4, 1)   # Base square
        ]
        
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
    
    def draw_3d_object(self):
        """Draw the current 3D object with futuristic styling"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glPushMatrix()
        
        # Apply transformations
        glTranslatef(self.position_x, self.position_y, self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)
        
        # Set neon color
        color = self.neon_colors[self.current_color]
        glColor3f(*color)
        
        # Draw selected shape
        if self.current_shape == 0:
            self.draw_cube_wireframe()
        elif self.current_shape == 1:
            self.draw_sphere_wireframe()
        elif self.current_shape == 2:
            self.draw_torus_wireframe()
        elif self.current_shape == 3:
            self.draw_pyramid_wireframe()
        
        # Add glow effect with larger transparent version
        glColor4f(color[0], color[1], color[2], 0.3)
        glPushMatrix()
        glScalef(1.1, 1.1, 1.1)
        if self.current_shape == 0:
            self.draw_cube_wireframe()
        elif self.current_shape == 1:
            self.draw_sphere_wireframe()
        elif self.current_shape == 2:
            self.draw_torus_wireframe()
        elif self.current_shape == 3:
            self.draw_pyramid_wireframe()
        glPopMatrix()
        
        glPopMatrix()
        
    def draw_ui(self, frame):
        """Draw UI elements on the camera frame"""
        # Add dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw UI text with futuristic style
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)  # Cyan
        
        shapes = ['CUBE', 'SPHERE', 'TORUS', 'PYRAMID']
        cv2.putText(frame, f'SHAPE: {shapes[self.current_shape]}', (10, 30), 
                    font, 0.7, color, 2)
        cv2.putText(frame, f'ZOOM: {abs(self.zoom):.1f}', (10, 60), 
                    font, 0.7, color, 2)
        cv2.putText(frame, 'Controls:', (10, 100), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, 'SPACE: Change Shape | C: Change Color', (10, 125), 
                    font, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("🚀 Futuristic 3D Hand Controller Started!")
        print("📋 Controls:")
        print("  • Two hands apart/together: ZOOM (smooth)")
        print("  • One hand open: ROTATE")
        print("  • One hand closed (fist or 2 fingers): MOVE (smooth)")
        print("  • SPACE: Change shape")
        print("  • C: Change color")
        print("  • Q or ESC: Quit")
        
        while running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.current_shape = (self.current_shape + 1) % 4
                    elif event.key == pygame.K_c:
                        self.current_color = (self.current_color + 1) % len(self.neon_colors)
            
            # Capture frame from camera
            success, frame = self.cap.read()
            if not success:
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )
            
            # Process gestures
            self.process_gestures(results)
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            # Show camera feed
            cv2.imshow('Hand Tracking', frame)
            
            # Draw 3D object
            self.draw_3d_object()
            pygame.display.flip()
            
            # Handle OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            
            clock.tick(60)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    controller = Hand3DController()
    controller.run()