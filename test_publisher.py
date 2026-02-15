
import zmq
def main():
    # 1. SETUP ZMQ PUBLISHER (to send commands to the robot)
    context = zmq.Context()
    cmd_pub_socket = context.socket(zmq.PUB)
    cmd_pub_socket.bind("tcp://*:5546") #


    print("\n" + "="*30)
    print(" ROBOT MONITOR CLI ACTIVE")
    print(" COMMANDS: reset, continue, stop, exit, status")
    print("="*30 + "\n")

    global running, stopped
    running = True
    try:
        while running:
            # Use input() to wait for user commands without blocking threads
            user_input = input(">> ").strip().lower()

            if user_input == "exit":
                print("Shutting down...")
                running = False
                break

            elif user_input == "reset":
                print("Sending manual RESET_POSE...")
                cmd_pub_socket.send_pyobj({"command": "RESET_POSE"})
            
            elif user_input == "continue":
                print("Sending manual CONTINUE...")
                stopped = False
                cmd_pub_socket.send_pyobj({"command": "CONTINUE"})

            elif user_input == "stop":
                print("Stopping monitor...")
                stopped = True
                cmd_pub_socket.send_pyobj({"command": "STOP"})

            elif user_input == "":
                continue

            else:
                # Fallback: send custom string commands directly as generic payloads
                print(f"Unknown command: {user_input}. Sending as generic command...")
                cmd_pub_socket.send_pyobj({"command": "CUSTOM", "value": user_input})

    except KeyboardInterrupt:
        running = False
        print("\nInterrupted by user.")

if __name__ == "__main__":
    main()