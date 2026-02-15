import zmq
import cv2
import numpy as np
import time
import re
import json
from PIL import Image
from google.api_core import retry 
import threading
import copy
from collections import deque
import google.genai as genai
from dotenv import load_dotenv
import os
from formula_to_dict_recovery import LTLMonitor
from datetime import datetime

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import time

@dataclass
class Keypoint:
    mode: str
    pose: np.ndarray          # joint configuration or EE pose
    timestamp: float
    grasped: int              # optional, but very useful

class KeypointStore:
    def __init__(self, reset_pose):
        self.keypoints: Dict[str, Keypoint] = {}
        self.reset_pose = reset_pose

    def save(self, mode, pose, grasped):
        self.keypoints[mode] = Keypoint(
            mode=mode,
            pose=np.array(pose, dtype=float),
            timestamp=time.time(),
            grasped=grasped
        )

    def has(self, mode):
        return mode in self.keypoints

    def get(self, mode) -> Optional[Keypoint]:
        return self.keypoints.get(mode, None)

    def get_reset_pose(self):
        return self.reset_pose


from enum import Enum

class RecoveryAction(Enum):
    RECOVER_TO_MODE = 1
    RETRY_PERCEPTION = 2
    ROLLBACK = 3
    ABORT = 4

def decide_recovery(
    inferred_mode,
    current_mode,
    retry_count,
    keypoint_store
):

    # Case 2: perception might be wrong
    if retry_count < 2:
        return (RecoveryAction.RETRY_PERCEPTION, None)
    
    if inferred_mode is None:
        ## reset yaane
        return (RecoveryAction.ROLLBACK, None)
    
    # # Case 1: predicates correspond to a valid mode
    # if inferred_mode is not None and inferred_mode != current_mode:
    #     print(f"inferred_mode: {inferred_mode}")
    #     return (RecoveryAction.RECOVER_TO_MODE, inferred_mode)

    # # Case 3: rollback possible
    # if keypoint_store.has(current_mode):
    #     print(f"keypoint: {keypoint_store}")
    #     return (RecoveryAction.ROLLBACK, current_mode)

    # # Case 4: nothing left
    # return (RecoveryAction.ABORT, None)


timestamp = datetime.now().strftime("%m_%d_%H_%M")
## get predicates from gemini
load_dotenv(dotenv_path="/home/olagh48652/task_monitor/.env/api_keys.env")
PREDICATE_POOL_PATH = "/home/olagh48652/task_monitor/ltl_llm_dec/robotics_CHECKLIST_pot_demo/demo_0_pool_of_predicates.json"
OUTPUT_FOLDER = f"/home/olagh48652/task_monitor/task_pot/inf_folders_sim/inf_pred_{timestamp}" ## save label outputs
api_keys_available = ["GEMINI_API_KEY", "GEMINI_API_KEY_LEEN", "GEMINI_API_KEY_NOUR", "GEMINI_API_KEY_HASSAN"]
api_keys_used = []


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
DEBUG_FOLDER = f"debug_prompts_{timestamp}"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# --- CONFIG ---
HISTORY_LEN = 3
running = True # Global flag to control threads

# --- SHARED RESOURCES ---
# The buffer is shared between the two threads
shared_buffer = deque(maxlen=HISTORY_LEN)
buffer_lock = threading.Lock()

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def create_dynamic_schema(predicate_pool):
    """
    Builds a strict JSON schema that forces the model to include 
    every predicate in the pool as a required key.
    """
    # 1. Structure for a single predicate result
    single_predicate_schema = {
        "type": "OBJECT",
        "properties": {
            "is_true": {"type": "BOOLEAN"},
            "evidence": {"type": "STRING"}
        },
        "required": ["is_true", "evidence"]
    }

    # 2. Force every predicate name to be a property
    predicate_properties = {
        pred_name: single_predicate_schema 
        for pred_name in predicate_pool
    }

    # 3. Final Schema Structure
    final_schema = {
        "type": "OBJECT",
        "properties": {
            "intervals": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "predicates": {
                            "type": "OBJECT",
                            "properties": predicate_properties,
                            "required": predicate_pool # <--- This enforces all predicates exist
                        }
                    },
                    "required": ["predicates"]
                }
            }
        },
        "required": ["intervals"]
    }
    
    return final_schema
    
def load_general_predicates(path):
    """
    Extract predicate names (e.g., reachable(bowl)) from a JSON-like file.
    Returns a sorted list of unique predicate strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    predicates = re.findall(r'"([^"]+)"', content)

    return sorted(set(predicates))

def fmt_list(data, precision=4):
    """
    Helper to round floats and compact lists for token efficiency.
    Input: [0.123456, 0.987654] -> Output String: "[0.1235, 0.9877]"
    """
    if data is None: return "[]"
    # Handle numpy arrays or lists
    if hasattr(data, 'tolist'): 
        data = data.tolist()
    
    # If it's a list of lists (like bounding boxes), handle recursively
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        return "[" + ",".join(fmt_list(sub, precision) for sub in data) + "]"
    
    # Round floats, keep ints as is
    formatted = []
    for x in data:
        if isinstance(x, float):
            formatted.append(f"{x:.{precision}f}")
        else:
            formatted.append(str(x))
            
    # Join with comma only (no space) to save 1 token per number
    return "[" + ",".join(formatted) + "]"

# ------------------------------------------------------------
# Parse Gemini output
# ------------------------------------------------------------
import json

def json_to_predicate_vector(json_data, predicate_to_idx, PRED_DIM):
    """
    Parses JSON intervals and returns a binary predicate vector.
    - Only predicates with is_true == True are included
    - Predicate names are sorted alphabetically before vectorization
    """

    pred_vector = np.zeros(PRED_DIM, dtype=int)

    intervals = json_data.get("intervals", [])
    print("len of interval: ", len(json_data.get("intervals", [])))
    
    if len(json_data.get("intervals", [])) != 1:
        print("WARN len of interval not one ")
    for interval in intervals:
        predicates_dict = interval.get("predicates", {})

        true_preds = sorted(
            p_name
            for p_name, p_data in predicates_dict.items()
            if p_data.get("is_true") is True
        )

        for p in true_preds:
            if p in predicate_to_idx:
                pred_vector[predicate_to_idx[p]] = 1

    return pred_vector
 
# ------------------------------------------------------------
# Gemini API Call Functions
# ------------------------------------------------------------
def is_retryable(e) -> bool:
    
    global client, GEMINI_API_KEY
    print("e: ", e)
    
    if retry.if_transient_error(e):
        print(" if_transient_error. Retrying...")
        return True  # catches standard transient network/API errors
    elif isinstance(e, genai.errors.ClientError) and e.code == 429:
        print(" Quota exceeded (429). Retrying...")
        msg = str(e)
        if "GenerateRequestsPerMinutePerProjectPerModel" in msg:
            # Per-minute rate limit
            print(" Per-minute rate limit hit (GenerateRequestsPerMinutePerProjectPerModel).")
            return True
        elif "GenerateRequestsPerDayPerProjectPerModel" in msg:
            # Daily quota
            print(" Daily quota exhausted. Rotating API key.")
            ## add key
            
            api_keys_used.append("GEMINI_API_KEY")
            for api_key in api_keys_available:
                if api_key not in api_keys_used:
                    GEMINI_API_KEY = os.environ[api_key]
                    print(f" Using default API key {api_key}")
                    api_keys_used.append(api_key)
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    return True
            
            print(" All API keys exhausted. Stopping retries.")
            raise RuntimeError("All Gemini API keys exhausted.")
        else:
            # Generic 429 fallback
            print("  Unknown 429 error, retrying after 30s.")
            time.sleep(30)
            return True
    elif isinstance(e, genai.errors.ServerError) and e.code == 503:
        print(" Model overloaded (503). Retrying...")
        return True  # service overloaded
    else:
        return False  # do not retry for other errors
    
def generate_with_retry(model_id, contents, gen_config):
    ## to reflect switched key
    
    @retry.Retry(predicate=is_retryable, initial=10, multiplier=1.5, deadline=300)
    def inner():
        global client
        return client.models.generate_content(model=model_id, contents=contents, config=gen_config)
    return inner()

# ------------------------------------------------------------
# Building prompts 
# ------------------------------------------------------------
def generate_prompt(predicate_pool):

    predicates_str = ", ".join(predicate_pool)
    
    return [f"""
### ROLE
You are a **Vision–Kinematics Predicate Grounding Analyst** for robotic manipulation
operating in **INFERENCE MODE under PARTIAL OBSERVABILITY**.

You must infer predicate truth values using:
- Current visual evidence
- Current robot state
- **Temporal history of observations AND past predicate decisions**

You are NOT guaranteed full visibility of all objects at the current timestep.

---

### OPERATING ASSUMPTION (CRITICAL)

You are running at **inference time**, not training or offline labeling.

This means:
- Objects may be **temporarily occluded**
- Objects may leave the camera view while still existing
- Visual absence alone is NEVER sufficient to set a predicate to False

You must explicitly reason over **temporal continuity**.

---

### INPUT DATA (SYNCHRONIZED PER TIMESTEP)

Each timestep may include:

#### 1. Robot State
- Joint States
- End-Effector (EE) Pose
- Gripper State

#### 2. Visual Data
- Raw Front View
- Wrist View

#### 3. Predicate History (WHEN PROVIDED)
- Predicate truth values inferred at previous timesteps

Predicate history represents the system’s **best known world state** unless contradicted.

---

### CORE PRINCIPLE (REFINED)

A predicate is **True** if:
- There is **positive visual evidence**, OR
- The predicate was **True in history** AND there is **no contradicting evidence**

A predicate is **False** only if:
- There is **explicit negative evidence** from vision or kinematics
- OR robot state makes the predicate physically impossible

**Absence of visual evidence ≠ evidence of absence.**

---

### TEMPORAL CONSISTENCY RULES (NEW — NON-NEGOTIABLE)

1. **Persistence Rule**
   - If a predicate was True at T-1, it remains True at T
     unless explicitly contradicted.

2. **Occlusion Rule**
   - If an object is not visible but was previously visible
     and no contradiction exists, assume it still exists.

3. **Contradiction Priority**
   - Explicit contradictions override history.
   - Examples:
     - Gripper opens → previously grasped object becomes ungrasped
     - EE pose moves far from container → inside(A,B) becomes False

4. **No Hallucinated State Changes**
   - Do NOT infer state changes without causal evidence
     (motion, contact, release, insertion, etc.)

---

### PREDICATES TO EVALUATE
{predicates_str}

---

## VISUAL–KINEMATIC DECISION RULES

### 1. `grasped(Object)`

True if:
- Gripper is closed or partially closed AND
- Object is visually between fingers OR
- Object was grasped in history and gripper has not opened

False if:
- Gripper is open
- OR object is clearly not in contact

---

### 2. `on(A, B)` vs `above(A, B)`

Use CURRENT vision when available.
If A or B is occluded:
- Defer to history unless contradicted by EE motion or physics.

IMPORTANT: Height, EE Z-position, or relative vertical ordering
is NEVER sufficient on its own.

A predicate MUST be False unless ALL required subchecks pass.

---

**Step 1 — Alignment Check (MANDATORY, FIRST)**
- Using the RAW FRONT VIEW:
  - Project a vertical line downward from the centroid of A.
  - Decide explicitly: DOES THIS LINE INTERSECT B? (Yes / No)

If the answer is **No**:
- `on(A,B) = False`
- `above(A,B) = False`
- STOP. Do not evaluate further.

---

**Step 2 — Vertical Relationship Check (Context Sensitive)**

**CASE 1: A is an OBJECT (e.g., block, meat, bottle)**
- Apply the **Drop Test**: If A were released at this exact moment, would it land DIRECTLY on B?
- If A would land on an intermediate object (e.g., a pot) instead of B, then `above(A, B) = False`.

**CASE 2: A is the ROBOT END-EFFECTOR (EE)**
- **Skip the Drop Test.**
- The EE is `above(B)` if it is physically higher than B and passes the Alignment Check (Step 1).
- **Ignore intermediate objects.** (e.g., If EE is over a pot, and the pot is on the burner, `above(EE, burner)` is **TRUE**).

---

**Step 3 — Contact Check**
- `on(A,B)` → aligned + NO visible air gap + direct physical contact; indirect or mediated support does not count.
- `above(A,B)` → aligned + VISIBLE air gap

Mutual exclusion enforced. So A,B cant be on and above at the same time.

---

### 3. `inside(A, B)`

True if:
- Visual enclosure OR
- Lower portion of A must be occluded by B’s walls.
- History indicates inside AND EE pose has not exited B

False only if:
- A is clearly outside B
- OR EE pose contradicts containment

---

### 4. Global Physical Consistency

- Open gripper ⇒ no object is grasped
- Do not infer disappearance without cause
- Do not infer support or containment from segmentation alone.

---

### OUTPUT INSTRUCTION

You must analyze the **entire temporal sequence**.
Use history to maintain state under occlusion.
Group consecutive timesteps with identical predicate assignments into intervals.

Output JSON only.

Now generate the interval report for the provided data.
"""]

def cv2_to_pil(cv_img):
    """Helper to convert OpenCV (BGR) images to PIL (RGB) for Gemini."""
    if cv_img is None: return None
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def format_history_prompt(history_buffer, last_belief=None):
    """
    Takes a deque of timestep data and flattens it into 
    Gemini format.
    """
    full_prompt_content = []
    
    if last_belief is not None:
        full_prompt_content.append(
            "\n--- PREVIOUS INFERENCE BELIEF (AUTHORITATIVE) ---\n"
        )
        full_prompt_content.append(
            f"At the last inference step, the predicate values were:\n"
            f"{json.dumps(last_belief, indent=2)}\n"
        )
        full_prompt_content.append(
            "Unless explicitly contradicted by current evidence, "
            "these predicates should persist.\n"
        )
        
    current_idx = len(history_buffer) - 1
    
    for i, step_data in enumerate(history_buffer):
        is_current = (i == current_idx)
        
        # dynamic header to help the model distinguish history from now
        if is_current:
            header = f"\n\n--- CURRENT TIMESTEP (T={step_data['t']}) ---\n"
        else:
            # Calculate relative time, e.g., T-2, T-1
            offset = current_idx - i
            header = f"\n\n--- HISTORY CONTEXT (T-{offset}) ---\n"

        full_prompt_content.append(header)
        
        # Add Text Data
        state_text = (
            f"Joints: {step_data['joints']}\n"
            f"EE Pose: {step_data['ee']}\n"
            f"Gripper: {step_data['gripper']}\n"
        )
        full_prompt_content.append(state_text)
        
        # Add Images (Convert to PIL if they aren't already)
        if step_data['front_img']:
            full_prompt_content.append("Front View:")
            full_prompt_content.append(step_data['front_img'])
            
        if step_data['wrist_img']:
            full_prompt_content.append("Wrist View:")
            full_prompt_content.append(step_data['wrist_img'])
            
    return full_prompt_content

# ------------------------------------------------------------
# Thread 1: The Receiver (Fast, ~30Hz+)
# ------------------------------------------------------------

def data_receiver_thread(port=5540):
    """
    Continuously reads from ZMQ and updates the shared buffer.
    """
    global shared_buffer, running, stopped, grasped
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.bind(f"tcp://*:{port}")
    socket.subscribe(b"")
    socket.setsockopt(zmq.CONFLATE, 1) # Keep only newest message in ZMQ queue

    print(f"Receiver Thread: Listening on {port}...")

    while running:
        try:
            # 1. Non-blocking check or blocking recv
            # Using blocking recv is fine here because this thread does nothing else
            
            # raw = socket.recv()   # NOT recv_pyobj
            # print("RAW TYPE:", type(raw))
            # print("RAW LEN:", len(raw))
            # print("RAW FIRST 32 BYTES:", raw[:32])
            # print(f"running {running}")
            message = socket.recv_pyobj() 
            # print(f"Received message with keys: {list(message.keys())}")

            # --- HANDLE RESET SIGNAL ---
            if 'event' in message and message['event'] == 'robot_reset':
                ts = message.get("timestep", "N/A")
                print(f"!!! ROBOT RESET DETECTED at T={ts} !!!")
                print(">>> Wiping History Buffer...")

                # 1. ACQUIRE LOCK 
                with buffer_lock:
                    shared_buffer.clear()
            
                # 2. Skip the rest of the loop for this message
                continue
            if 'event' in message and message['event'] == 'robot_continue':
                ## clear buffer
                # print(f"!!! ROBOT RESET DETECTED at T={ts} !!!")
                print(">>> Wiping History Buffer...")
                
                with buffer_lock:
                    shared_buffer.clear()
                stopped = False
                continue
            
            # 2. Process Images (Decode from Bytes)
            img0_pil, img1_pil = None, None
            
            if 'cam0_jpg' in message:
                arr = np.frombuffer(message['cam0_jpg'], np.uint8)
                img0_pil = Image.fromarray(cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            
            if 'cam1_jpg' in message:
                arr = np.frombuffer(message['cam1_jpg'], np.uint8)
                img1_pil = Image.fromarray(cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

            
            raw_gripper = message.get('gripper_state', None)
            # print("raw_gripper: ", raw_gripper)
            grasped_scalar = int(raw_gripper[0] <= 0.05) if raw_gripper is not None else 0
            
            # 3. Format Data
            # Note: We do formatting here to save time in the main thread
            step_data = {
                't': time.time(), # or message index
                'joints': fmt_list(message.get('q', [])),
                'ee': fmt_list(message.get('ee_pos', [])),
                'gripper': fmt_list(raw_gripper),
                'grasped': grasped_scalar, 
                'front_img': img0_pil,
                'wrist_img': img1_pil
            }
            

            # 4. CRITICAL: LOCK AND UPDATE
            with buffer_lock:
                shared_buffer.append(step_data)
                
        except Exception as e:
            print(f"Receiver Error: {e}")
            time.sleep(0.1)

    socket.close()
    context.term()
    print("Receiver Thread Stopped.")

# ------------------------------------------------------------
# Thread 3: Visualization
# ----------------------------------------------------   
# Global shared buffer for the visualization thread
vis_image_buffer = [] 
vis_lock = threading.Lock()

def visualization_worker():
    """
    Runs in a separate thread. continuously renders whatever is 
    in 'vis_image_buffer' so the window stays responsive.
    """
    global running, vis_image_buffer
    
    print("Visualization Thread: Started.")
    VIS_HEIGHT = 300
    window_name = "VLM Context View"
    
    while running:
        # 1. Get the latest images safely
        current_images = []
        with vis_lock:
            if vis_image_buffer:
                current_images = list(vis_image_buffer)
        
        # 2. Render if we have images
        if current_images:
            try:
                processed_imgs = []
                for img in current_images:
                    # Resize for consistency
                    h, w, _ = img.shape
                    aspect_ratio = w / h
                    new_w = int(VIS_HEIGHT * aspect_ratio)
                    resized = cv2.resize(img, (new_w, VIS_HEIGHT))
                    processed_imgs.append(resized)
                
                # Stack horizontally
                if processed_imgs:
                    combined = np.hstack(processed_imgs)
                    cv2.imshow(window_name, combined)
            except Exception as e:
                print(f"Vis Thread Error: {e}")
        
        # 3. CRITICAL: This keeps the window alive. 
        # It runs every 30ms, independent of the Gemini API call in the main thread.
        cv2.waitKey(30)
        
    print("Visualization Thread: Stopping.")
    cv2.destroyAllWindows()

# ------------------------------------------------------------
# Thread 2: The Logic/LLM (Slow, ~0.5 - 1Hz)
# ------------------------------------------------------------
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# --- WORKER FUNCTION ---
# This runs in the background. We pass 'grasped_state' in so it travels with the request.
def call_gemini_worker(client, model_id, payload, config, step_id, grasped_state, pose):
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=payload,
            config=config
        )
        
        # Parse JSON here to offload work from main thread
        text_response = response.text
        try:
            parsed_json = json.loads(text_response)
        except:
            parsed_json = None

        return {
            "step_id": step_id,
            "json_output": parsed_json,  # Return parsed object directly
            "raw_text": text_response,   # Keep raw text just in case
            "grasped_state": grasped_state, 
            "pose": pose, 
            "error": None
        }
    except Exception as e:
        return {
            "step_id": step_id,
            "json_output": None,
            "raw_text": None,
            "grasped_state": grasped_state,
            "pose": pose, 
            "error": str(e)
        }

def trigger_perception_standalone():
    global shared_buffer, running, stopped, last_inferred_predicates, last_inference_time, vis_image_buffer
    global step_counter
    predicate_pool = load_general_predicates(PREDICATE_POOL_PATH)
    system_instruction = generate_prompt(predicate_pool)
    strict_schema = create_dynamic_schema(predicate_pool)
    
    gen_config = {
        "response_mime_type": "application/json",
        "response_schema": strict_schema
    }
    
    PRED_DIM = len(predicate_pool)
    predicate_to_idx = {p: i for i, p in enumerate(predicate_pool)}
    
    with buffer_lock:
        current_context_snapshot = list(shared_buffer)
    last_trigger_time = time.time() # Reset timer
    
    last_inferred_predicates = None
    # Prepare Prompt (Main thread does this quickly)
    last_step = current_context_snapshot[-1]
    history_payload = format_history_prompt(current_context_snapshot, last_belief=last_inferred_predicates)
    final_payload = [system_instruction] + history_payload
    
    current_grasped_state = last_step.get('grasped', 0)
    current_pose = last_step.get('joints', 0)
    print("current_grasped_state: ", current_grasped_state)
    print("current_pose: ", current_pose)
    
    # Update Visualization 
    try:
        pil_images = [step_data['front_img'] for step_data in current_context_snapshot]
        if pil_images:
            cv_converted = []
            for p_img in pil_images:
                if p_img.mode != 'RGB': p_img = p_img.convert('RGB')
                arr = np.array(p_img)[:, :, ::-1].copy()
                cv_converted.append(arr)
            with vis_lock:
                vis_image_buffer = cv_converted
    except Exception as e:
        print(f"Vis error: {e}")
        
    response = generate_with_retry(MODEL_ID, final_payload, gen_config)
    if response.parts:
        json_output = json.loads(response.text)
        
        last_inferred_predicates = json_output
        last_inference_time = time.time()
        
        output_file = os.path.join(OUTPUT_FOLDER, f"{step_counter}_true_predicates.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=4)
            
        step_counter += 1  
        print("Successfully parsed JSON.")            
        
        ## if we get a response 
        pred_vector = json_to_predicate_vector(
            json_output,
            predicate_to_idx,
            PRED_DIM
        )
        ## since gthe last measuremtn is from proprioceptive data
        print("pred_vector: ",pred_vector)
        pred_vector = np.append(pred_vector,  last_step['grasped'])
        print("pred_vector after appending: ",pred_vector)
        # update the predicates
        ltl_monitor.predicates = pred_vector
        # run a step
        status = ltl_monitor.monitor_step()
        print("LTL Monitor Status: ", status)
        if status == "violation":
            print("!!! LTL VIOLATION — SENDING STOP TO ROBOT !!!")
        elif status == "done":
            print("!!! LTL MONITOR: TASK DONE DETECTED !!!")
        return  status      
        
import collections           
def llm_monitor_loop_sim(ltl_monitor, cmd_pub_socket):
    global shared_buffer, running, stopped, last_inferred_predicates, last_inference_time, vis_image_buffer
    global client # We will replace single client with a list
    global step_counter
    print("LLM Monitor: Started (Parallel Mode).")
    
    # --- 1. SETUP MULTI-KEY ROTATION ---
    # Load all keys
    keys = [
        os.environ.get("GEMINI_API_KEY_HASSAN"),
        os.environ.get("GEMINI_API_KEY_NOUR"),
        os.environ.get("GEMINI_API_KEY_LYNN"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_GMAIL"),
    ]
    # Filter out None keys if environment vars are missing
    keys = [k for k in keys if k]
    
    # Create a client for each key
    clients = [genai.Client(api_key=k) for k in keys]
    client_idx = 0
    print(f"Loaded {len(clients)} API clients for rotation.")

    # --- 2. THREAD POOL SETUP ---
    # Max workers = number of keys (or more, but keys is usually the bottleneck)
    executor = ThreadPoolExecutor(max_workers=len(clients))
    
    # This deque will store tuples: (step_id, future_object)
    # We push to the right, and peek/pop from the left (oldest).
    pending_tasks = collections.deque()
    
    # --- 3. STANDARD SETUP ---
    vis_thread = threading.Thread(target=visualization_worker, daemon=True)
    vis_thread.start()
    
    predicate_pool = load_general_predicates(PREDICATE_POOL_PATH)
    system_instruction = generate_prompt(predicate_pool)
    strict_schema = create_dynamic_schema(predicate_pool)
    
    gen_config = {
        "response_mime_type": "application/json",
        "response_schema": strict_schema
    }

    step_counter = 0
    stopped = False
    PRED_DIM = len(predicate_pool)
    predicate_to_idx = {p: i for i, p in enumerate(predicate_pool)}
    last_inferred_predicates = None
    # Timer for triggering requests
    last_trigger_time = 0
    TRIGGER_INTERVAL = 15.0  # Seconds between launching new inference requests

    keypoints = KeypointStore(reset_pose=np.zeros(7))
    violation_retry_count = 0

    while running:
        if stopped:
            time.sleep(1)
            continue
            
        current_time = time.time()
        
        # ==========================================================
        # PHASE A: SUBMISSION (Is it time to launch a new request?)
        # ==========================================================
        if current_time - last_trigger_time >= TRIGGER_INTERVAL:
            
            # 1. Get Snapshot
            current_context_snapshot = []
            with buffer_lock:
                if len(shared_buffer) > 0:
                    current_context_snapshot = list(shared_buffer)
                else:
                    print("shared buffer less than 1...")
            
            if current_context_snapshot:
                last_trigger_time = current_time # Reset timer
                
                # Prepare Prompt (Main thread does this quickly)
                last_step = current_context_snapshot[-1]
                
                history_payload = format_history_prompt(current_context_snapshot, last_belief=last_inferred_predicates)
                final_payload = [system_instruction] + history_payload
                
                current_grasped_state = last_step.get('grasped', 0)
                current_pose = last_step.get('joints', 0)
                print("current_grasped_state: ", current_grasped_state)
                print("current_pose: ", current_pose)
                
                # Update Visualization 
                try:
                    pil_images = [step_data['front_img'] for step_data in current_context_snapshot]
                    if pil_images:
                        cv_converted = []
                        for p_img in pil_images:
                            if p_img.mode != 'RGB': p_img = p_img.convert('RGB')
                            arr = np.array(p_img)[:, :, ::-1].copy()
                            cv_converted.append(arr)
                        with vis_lock:
                            vis_image_buffer = cv_converted
                except Exception as e:
                    print(f"Vis error: {e}")

                # Select Client (Round Robin)
                active_client = clients[client_idx]
                client_idx = (client_idx + 1) % len(clients)

                print(f"--- Submitting Step {step_counter} to background (Key #{client_idx}) ---")
                
                # SUBMIT TO THREAD POOL
                # We pass 'step_counter' so the result knows which step it belongs to
                future = executor.submit(
                    call_gemini_worker, 
                    active_client, 
                    MODEL_ID, 
                    final_payload, 
                    gen_config, 
                    step_counter,
                    current_grasped_state,
                    current_pose,
                )
                
                # Store in our ordered queue
                pending_tasks.append(future)
                
                step_counter += 1
            else:
                # Buffer empty, retry soon
                time.sleep(0.1)
                print("buffer empty retrying")

        # ==========================================================
        # PHASE B: PROCESSING (Check strictly in order)
        # ==========================================================
        
        # Look at the OLDEST task (index 0)
        if pending_tasks:
            oldest_future = pending_tasks[0]
            
            # check if it is done (non-blocking check)
            if oldest_future.done():
                # Remove from queue
                pending_tasks.popleft()
                
                # Get result
                result = oldest_future.result()
                r_step_id = result["step_id"]
                r_json = result["json_output"]
                r_error = result["error"]
                r_grasped = result["grasped_state"]
                r_pose = result["pose"]
                
                print(f"Processing result for Step {r_step_id}...")

                if r_error:
                    print(f"API Error on step {r_step_id}: {r_error}")
                    # Decide how to handle error: Skip? Stop? Retry?
                    # usually skip updates to avoid crashing LTL
                
                elif r_json:
                    # print(f"r_json : {r_json}")
                    try:
                        last_inferred_predicates = r_json
                        last_inference_time = time.time()
                        
                        # Save logs (Keep your existing logic)
                        output_file = os.path.join(OUTPUT_FOLDER, f"{r_step_id}_true_predicates.json")
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(r_json, f, indent=4)

                        pred_vector = json_to_predicate_vector(r_json, predicate_to_idx, PRED_DIM)
                        r_json = None
                        
                        # ... Assuming you fix the grasped logic ...
                        pred_vector = np.append(pred_vector, r_grasped) 

                        ltl_monitor.predicates = pred_vector
                        result = ltl_monitor.monitor_step()
                        print("result ",result)
                        print(f"Step {r_step_id} LTL Status: {result.status}")

                        if result.status == "violation":
                            print("!!! VIOLATION DETECTED !!!")
                            stop_payload = {"command": "STOP", "reason": "LTL violation"}
                            cmd_pub_socket.send_pyobj(stop_payload)
                            stopped = True
                            
                            print(f"keypoint: {keypoints}")
                            
                            # Cancel all other pending tasks to save money/computation
                            for f in pending_tasks:
                                f.cancel()
                            pending_tasks.clear()
                            
                            inferred_mode = ltl_monitor.infer_mode_from_predicates(pred_vector)
                            print(f"inferred_mode: {inferred_mode}")
                            
                            print(f"pose: {r_pose}")
                            print(f"type pose: {type(r_pose)}")
                            
                            action, target = decide_recovery(
                                inferred_mode=inferred_mode,
                                current_mode=result.curr_mode,
                                retry_count=violation_retry_count,
                                keypoint_store=keypoints
                            )
                            
                            print(f" action: {action}")

                            # violation_retry_count += 1

                            ## TODO TEST WHERE A GOOD PLACEMENT FOR THIS IS.
                            # # Cancel all other pending tasks to save money/computation
                            # for f in pending_tasks:
                            #     f.cancel()
                            # pending_tasks.clear()
                            
                            if action == RecoveryAction.RECOVER_TO_MODE:
                                print(f"Recovering to mode {target}")
                                kp = keypoints.get(target)
                                # cmd_pub_socket.send_pyobj({
                                #     "command": "RECOVER",
                                #     "mode": target,
                                #     "pose": kp.pose.tolist() if kp else None
                                # })
                                # stopped = False
                                # violation_retry_count = 0

                            elif action == RecoveryAction.RETRY_PERCEPTION:
                                print("Retrying perception")
                                new_stat = trigger_perception_standalone()
                                
                                if new_stat == "running":
                                    stopped = False 
                                    cmd_pub_socket.send_pyobj({"command": "CONTINUE"})

                            elif action == RecoveryAction.ROLLBACK:
                                print("Rolling back")
                                # kp = keypoints.get(result.curr_mode)
                                # cmd_pub_socket.send_pyobj({
                                #     "command": "ROLLBACK",
                                #     "pose": kp.pose.tolist()
                                # })

                            # elif action == RecoveryAction.ABORT:
                            #     print("Task not achievable")
                            #     cmd_pub_socket.send_pyobj({
                            #         "command": "ABORT",
                            #         "reason": "Unrecoverable violation"
                            #     })
                            #     break
                            
                        ## update keypoint after a successful transition
                        if result.status == "running" and result.next_mode != result.curr_mode:
                            keypoints.save(
                                mode=result.next_mode,
                                pose=r_pose,
                                grasped=r_grasped
                            )
    
                        elif result.status == "done":
                            print("!!! TASK DONE !!!")
                            done_payload = {"command": "TASK_DONE", "reason": "Success"}
                            cmd_pub_socket.send_pyobj(done_payload)
                            stopped = True
                            break

                    except Exception as e:
                        print(f"Error parsing/updating LTL for step {r_step_id}: {e}")
        
        # Small sleep to prevent CPU spinning while waiting for IO or timers
        time.sleep(0.05)
                   
import threading
import tkinter as tk

def gui_backdoor_buttons(ltl_monitor):
    global system_instruction, client, executor
    """
    Tkinter GUI with 4 buttons:
    - Get current predicates
    - Get current mode
    - Set predicates (update_mode=False)
    - Set predicates (update_mode=True)
    """
    root = tk.Tk()
    root.title("LTL Monitor Backdoor")

    # Entry for predicate vector input
    tk.Label(root, text="Predicate vector (0/1):").pack(padx=10, pady=5)
    entry = tk.Entry(root, width=50)
    entry.pack(padx=10, pady=5)

    # Status label
    status_label = tk.Label(root, text="Status: Idle")
    status_label.pack(padx=10, pady=5)

    def parse_vector():
        val = entry.get().strip()

        try:
            # Only allow 0, 1, or -
            if not all(c in {"0", "1", "-"} for c in val):
                raise ValueError("Only characters allowed: 0, 1, '-'")

            # Convert '-' → 0
            pred_vector = [0 if c == "-" else int(c) for c in val]

            # Optional but strongly recommended: length check
            if hasattr(ltl_monitor, "env_APs"):
                expected_len = len(ltl_monitor.env_APs)
                if len(pred_vector) != expected_len:
                    raise ValueError(
                        f"Expected {expected_len} predicates, got {len(pred_vector)}"
                    )

            print("[Backdoor] Parsed pattern vector:", pred_vector)
            return pred_vector

        except Exception as e:
            status_label.config(text=f"Invalid pattern: {e}")
            return None

    def get_preds():
        preds = ltl_monitor.get_current_predicates()
        print("[Backdoor] Current predicates:", preds)
        status_label.config(text=f"Printed predicates in console")

    def get_mode():
        mode = ltl_monitor.get_current_mode()
        print("[Backdoor] Current mode:", mode)
        status_label.config(text=f"Printed mode in console")

    def set_preds_no_update():
        pred_vector = parse_vector()
        if pred_vector is not None:
            target = ltl_monitor.get_mode_from_predicates(pred_vector)
            status_label.config(text=f"Set predicates (no mode update). Target={target}")

    def set_preds_update():
        pred_vector = parse_vector()
        if pred_vector is not None:
            success = ltl_monitor.set_predicates(pred_vector, update_mode=True)
            status_label.config(text=f"Set predicates (mode updated). Success={success}")
            
    def trigger_perception():
        global shared_buffer, running, stopped, last_inferred_predicates, last_inference_time, vis_image_buffer
        global step_counter
        predicate_pool = load_general_predicates(PREDICATE_POOL_PATH)
        system_instruction = generate_prompt(predicate_pool)
        strict_schema = create_dynamic_schema(predicate_pool)
        gen_config = {
            "response_mime_type": "application/json",
            "response_schema": strict_schema
        }
        
        PRED_DIM = len(predicate_pool)
        predicate_to_idx = {p: i for i, p in enumerate(predicate_pool)}
        
        with buffer_lock:
            current_context_snapshot = list(shared_buffer)
        last_trigger_time = time.time() # Reset timer
        
        last_inferred_predicates = None
        # Prepare Prompt (Main thread does this quickly)
        last_step = current_context_snapshot[-1]
        history_payload = format_history_prompt(current_context_snapshot, last_belief=last_inferred_predicates)
        final_payload = [system_instruction] + history_payload
        
        current_grasped_state = last_step.get('grasped', 0)
        current_pose = last_step.get('joints', 0)
        print("current_grasped_state: ", current_grasped_state)
        print("current_pose: ", current_pose)
        
        # Update Visualization 
        try:
            pil_images = [step_data['front_img'] for step_data in current_context_snapshot]
            if pil_images:
                cv_converted = []
                for p_img in pil_images:
                    if p_img.mode != 'RGB': p_img = p_img.convert('RGB')
                    arr = np.array(p_img)[:, :, ::-1].copy()
                    cv_converted.append(arr)
                with vis_lock:
                    vis_image_buffer = cv_converted
        except Exception as e:
            print(f"Vis error: {e}")
            
        response = generate_with_retry(MODEL_ID, final_payload, gen_config)
        if response.parts:
            json_output = json.loads(response.text)
            
            last_inferred_predicates = json_output
            last_inference_time = time.time()
            
            output_file = os.path.join(OUTPUT_FOLDER, f"{step_counter}_true_predicates.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=4)
                
            step_counter += 1  
                  
            print("Successfully parsed JSON.")            
            
            ## if we get a response 
            pred_vector = json_to_predicate_vector(
                json_output,
                predicate_to_idx,
                PRED_DIM
            )
            ## since gthe last measuremtn is from proprioceptive data
            print("pred_vector: ",pred_vector)
            pred_vector = np.append(pred_vector,  last_step['grasped'])
            print("pred_vector after appending: ",pred_vector)
            # update the predicates
            ltl_monitor.predicates = pred_vector
            # run a step
            status = ltl_monitor.monitor_step()
            print("LTL Monitor Status: ", status)
            if status == "violation":
                print("!!! LTL VIOLATION — SENDING STOP TO ROBOT !!!")
            elif status == "done":
                print("!!! LTL MONITOR: TASK DONE DETECTED !!!")

            

    # Buttons
    tk.Button(root, text="Get Current Predicates", command=get_preds).pack(padx=5, pady=5, fill='x')
    tk.Button(root, text="Get Current Mode", command=get_mode).pack(padx=5, pady=5, fill='x')
    tk.Button(root, text="Set Predicates (no mode update)", command=set_preds_no_update).pack(padx=5, pady=5, fill='x')
    tk.Button(root, text="Set Predicates (update mode)", command=set_preds_update).pack(padx=5, pady=5, fill='x')
    tk.Button(root, text="Trigger Perception", command=trigger_perception).pack(padx=5, pady=5, fill='x')

    root.mainloop()
   
# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    global client, GEMINI_API_KEY, MODEL_ID
    # api_keys_available = ["GEMINI_API_KEY", "GEMINI_API_KEY_LEEN", "GEMINI_API_KEY_NOUR", "GEMINI_API_KEY_HASSAN"]

    MODEL_ID = "gemini-3-flash-preview"
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY_NOUR"]
    # api_keys_used.append("GEMINI_API_KEY")
    # api_keys_used.append("GEMINI_API_KEY_HASSAN")
    # api_keys_used.append("GEMINI_API_KEY_NOUR") ## already done
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # --- COMMAND PUBLISHER (Monitor -> Robot) ---
    cmd_context = zmq.Context()
    cmd_pub_socket = cmd_context.socket(zmq.PUB)

    # IMPORTANT: bind here, robot connects
    cmd_pub_socket.bind("tcp://*:5546")

    # Give ZMQ time to establish the connection
    time.sleep(0.5)
    
    # 1. Start Receiver in Background
    t_recv = threading.Thread(target=data_receiver_thread, daemon=True)
    t_recv.start()

    ltl_monitor = LTLMonitor()
    
    # Start the GUI backdoor in a separate thread
    gui_thread = threading.Thread(target=gui_backdoor_buttons, args=(ltl_monitor,), daemon=True)
    gui_thread.start()
    try:
        # Give receiver a moment to warm up
        time.sleep(1) 
        llm_monitor_loop_sim(ltl_monitor, cmd_pub_socket)
    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        t_recv.join()
  
# {'above(black_pot, table)': 0,
#  'above(green_bell_pepper, table)': 1,
#  'above(lid, table)': 2,
#  'above(robot_endeffector, black_pot)': 3,
#  'above(robot_endeffector, green_bell_pepper)': 4,
#  'above(robot_endeffector, lid)': 5,
#  'above(robot_endeffector, single_burner)': 6,
#  'grasped(black_pot)': 7,
#  'grasped(green_bell_pepper)': 8,
#  'grasped(lid)': 9,
#  'inside(green_bell_pepper, black_pot)': 10,
#  'on(black_pot, single_burner)': 11,
#  'on(black_pot, table)': 12,
#  'on(green_bell_pepper, table)': 13,
#  'on(lid, black_pot)': 14,
#  'on(lid, table)': 15,
#  'on(single_burner, table)': 16}

# ah
# 000000001001010111

# aq
# 000000001000010111
# 000000000011000110
# python3 formula_to_dict.py 
# env_APs: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
# robot_APs: ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at']
# mode: aa

# 000000000001010110
# transitions: {'000----00000110110': 'aa', '-001---10000010111': 'ar', '0001---10000110111': 'af', '100000-10000010111': 'am'}
#   pattern: 000----00000110110  target: aa
#   pattern: -001---10000010111  target: ar
#   pattern: 0001---10000110111  target: af
#   pattern: 100000-10000010111  target: am
# mode: ab                       
# transitions: {'-00111-10001010111': 'ab', '-00001-00001010110': 'ag', '-0001-000001010110': 'ad'}
#   pattern: -00111-10001010111  target: ab
#   pattern: -00001-00001010110  target: ag
#   pattern: -0001-000001010110  target: ad
# mode: ac
# transitions: {}
# mode: ad
# transitions: {'-0001-000001010110': 'ad', '--0----01001000111': 'aq', '-000-0001001010111': 'ah', '--0--0-01011000111': 'an'}
#   pattern: -0001-000001010110  target: ad
#   pattern: --0----01001000111  target: aq
#   pattern: -000-0001001010111  target: ah
#   pattern: --0--0-01011000111  target: an
# mode: ae
# transitions: {'-------00011001010': 'al', '-------00111000011': 'ae', '-------00111001011': 'ap'}
#   pattern: -------00011001010  target: al
#   pattern: -------00111000011  target: ae
#   pattern: -------00111001011  target: ap
# mode: af
# transitions: {'-001---10000010111': 'ar', '0001---10000110111': 'af', '100000-10000010111': 'am'}
#   pattern: -001---10000010111  target: ar
#   pattern: 0001---10000110111  target: af
#   pattern: 100000-10000010111  target: am
# mode: ag
# transitions: {'-00001-00001010110': 'ag', '-0001-000001010110': 'ad'}
#   pattern: -00001-00001010110  target: ag
#   pattern: -0001-000001010110  target: ad
# mode: ah
# transitions: {'--0----01001000111': 'aq', '-000-0001001010111': 'ah'}
#   pattern: --0----01001000111  target: aq
#   pattern: -000-0001001010111  target: ah
# mode: ai
# transitions: {'--0----00011000110': 'ai', '-------00111000011': 'ae', '--0001000111000111': 'aj'}
#   pattern: --0----00011000110  target: ai
#   pattern: -------00111000011  target: ae
#   pattern: --0001000111000111  target: aj
# mode: aj
# transitions: {'-------00111000011': 'ae', '--0001000111000111': 'aj'}
#   pattern: -------00111000011  target: ae
#   pattern: --0001000111000111  target: aj
# mode: ak
# transitions: {}
# mode: al
# transitions: {'-------00011001010': 'al'}
#   pattern: -------00011001010  target: al
# mode: am
# transitions: {'-00000-10001010111': 'ao', '100000-10000010111': 'am', '-00001-00001010110': 'ag'}
#   pattern: -00000-10001010111  target: ao
#   pattern: 100000-10000010111  target: am
#   pattern: -00001-00001010110  target: ag
# mode: an
# transitions: {'--0--0-01011000111': 'an', '--0----00011000110': 'ai'}
#   pattern: --0--0-01011000111  target: an
#   pattern: --0----00011000110  target: ai
# mode: ao
# transitions: {'-00000-10001010111': 'ao', '-00001-00001010110': 'ag'}
#   pattern: -00000-10001010111  target: ao
#   pattern: -00001-00001010110  target: ag
# mode: ap
# transitions: {'-------00011001010': 'al', '-------00111001011': 'ap'}
#   pattern: -------00011001010  target: al
#   pattern: -------00111001011  target: ap
# mode: aq
# transitions: {'--0----01001000111': 'aq', '--0--0-01011000111': 'an', '--0----00011000110': 'ai'}
#   pattern: --0----01001000111  target: aq
#   pattern: --0--0-01011000111  target: an
#   pattern: --0----00011000110  target: ai
# mode: ar
# transitions: {'-001---10000010111': 'ar', '-00111-10001010111': 'ab'}
#   pattern: -001---10000010111  target: ar
#   pattern: -00111-10001010111  target: ab
# mode: as
# transitions: {}
# mode: at
# transitions: {}
# done
