import zmq
import cv2
import numpy as np
import time

def main(port=5540):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # Using connect since your simulation/robot script is likely the binder
    socket.connect(f"tcp://localhost:{port}")
    socket.subscribe(b"")
    socket.setsockopt(zmq.CONFLATE, 1) 

    print(f"Subscribed to port {port}. Waiting for data...")
    print("Click the image window and press 'q' to quit.")

    try:
        while True:
            # 1. Receive the message object
            message = socket.recv_pyobj()

            # 2. Extract and Print Kinematics
            q = message.get('q', [])
            ee = message.get('ee_pos', [])
            gripper = message.get('gripper_state', [0.0])

            # Print to terminal for testing
            print(f"Joints: {np.round(q, 2)} | EE: {np.round(ee, 2)} | Grip: {gripper[0]:.3f}")

            # 3. Decode and Show Images
            # We check both cam0 and cam1 as your original script uses both
            imgs_to_show = []

            for cam_key in ['cam0_jpg', 'cam1_jpg']:
                if cam_key in message:
                    # Convert bytes to numpy array
                    arr = np.frombuffer(message[cam_key], np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        imgs_to_show.append(img)

            # 4. Display Logic
            if imgs_to_show:
                # Stack images horizontally if both exist, otherwise show one
                combined = np.hstack(imgs_to_show) if len(imgs_to_show) > 1 else imgs_to_show[0]
                
                cv2.imshow("Robot Testing Feed", combined)

            # --- CRITICAL: This is what makes the image actually show up ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        socket.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()