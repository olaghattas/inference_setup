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
from formula_to_dict_recovery_rule import LTLMonitor
from datetime import datetime

from dataclasses import dataclass
from typing import Dict, Optional

from enum import Enum

from rule_enforcer import LogicEnforcer

class ExecVisualState(Enum):
    RUNNING = "RUNNING"
    VIOLATION = "VIOLATION"
    RECOVERING = "RECOVERING"
    CONTINUING = "CONTINUING"
    ABORT = "ABORT"
    DONE = "DONE"
  
exec_visual_state = ExecVisualState.RUNNING
exec_visual_lock = threading.Lock()  
    
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
        pose_arr = np.array(pose, dtype=float)
        self.keypoints[mode] = Keypoint(
            mode=mode,
            pose=pose_arr,
            timestamp=time.time(),
            grasped=grasped
        )
        print(f"[Keypoint] Registered mode={mode} pose={pose_arr.tolist()} grasped={grasped}")

    def has(self, mode):
        return mode in self.keypoints

    def get(self, mode) -> Optional[Keypoint]:
        return self.keypoints.get(mode, None)

    def get_reset_pose(self):
        return self.reset_pose



class RecoveryAction(Enum):
    RECOVER_TO_MODE = 1 ## get current state after violation not none, if exists and has keypoint go to keypoint else go to reset pose
    RETRY_PERCEPTION = 2 ## retry perception, if its still violated open gripper 
    RECOVER_GET_STATE= 3 ## if state NONE, go to reset and get the state there
    ABORT = 4 ## go reset state and then run perception and still no state
    EMPTY_GRASP = 5 # open gripper if last is 1 and all grasped pred are 0


def decide_recovery(
    inferred_mode,
    current_mode,
    retry_count,
    keypoint_store,
    last_recovery,
):  
    # last_recovery should be wiped every time the policy is run 
    if retry_count < 1:
        return (RecoveryAction.RETRY_PERCEPTION, None)
    
    # Recovery already tried get-state-at-reset and we're still in violation → abort (no state satisfied)
    if last_recovery == RecoveryAction.RECOVER_GET_STATE:
        return (RecoveryAction.ABORT, None)

    if inferred_mode:
        
        return (RecoveryAction.RECOVER_TO_MODE, inferred_mode)
    else: ## not in keypoints so go to reset location
        return (RecoveryAction.RECOVER_GET_STATE, "reset_position")
    
    
    # # Case 1: predicates correspond to a valid mode
    # if inferred_mode is not None and inferred_mode != current_mode:
    #     return (RecoveryAction.RECOVER_TO_MODE, inferred_mode)

    # # Case 2: perception might be wrong
    # if retry_count < 2:
    #     return (RecoveryAction.RETRY_PERCEPTION, None)

    # # Case 3: rollback possible
    # if keypoint_store.has(current_mode):
    #     return (RecoveryAction.ROLLBACK, current_mode)

    # # Case 4: nothing left
    # return (RecoveryAction.ABORT, None)


timestamp = datetime.now().strftime("%m_%d_%H_%M")
## get predicates from gemini
load_dotenv(dotenv_path="/home/olagh48652/task_monitor/.env/api_keys.env")
PREDICATE_POOL_PATH = "/home/olagh48652/task_monitor/ltl_llm_dec/robotics_CHECKLIST_pot_demo/demo_0_pool_of_predicates.json"
OUTPUT_FOLDER = f"/home/olagh48652/task_monitor/inference_setup/inf_folders_sim/inf_pred_{timestamp}" ## save label outputs

api_keys_available = ["GEMINI_API_KEY", "GEMINI_API_KEY_LEEN", "GEMINI_API_KEY_NOUR", "GEMINI_API_KEY_HASSAN", "GEMINI_API_KEY_GMAIL"]
api_keys_used = []

print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# DEBUG_FOLDER = f"debug_prompts_{timestamp}"
# os.makedirs(DEBUG_FOLDER, exist_ok=True)

# --- CONFIG ---
HISTORY_LEN = 3
running = True # Global flag to control threads

# --- SPEED / RELIABILITY NOTES ---
# - Too slow / violations detected too late: (1) Lower TRIGGER_INTERVAL so inference runs more often.
#   (2) Use a faster model (e.g. gemini-2.5-flash) for lower latency; trade-off is more false preds.
# - Flash gives many false preds / violations only after catastrophe: (1) Prefer gemini-2.5-pro for
#   accuracy when latency allows. (2) Keep HISTORY_LEN and initial_step so the model has temporal
#   context. (3) Use perception_retry and recovery loop so one bad inference doesn't stop the task.
# - To debug: check OUTPUT_FOLDER for {step_id}_front.jpg, {step_id}_wrist.jpg next to
#   {step_id}_true_predicates.json to see exactly what the robot saw at each inference.

# Step-through mode: when enabled, pause at recovery/continue points until "Next Step" is clicked in GUI
step_mode_enabled = True   # If True, pause at step points; if False, run without pausing
step_mode = True           # True = currently paused (waiting for Next); GUI sets False to release
step_mode_lock = threading.Lock()

# --- SHARED RESOURCES ---
# The buffer is shared between the two threads
shared_buffer = deque(maxlen=HISTORY_LEN)
buffer_lock = threading.Lock()
# Step-sync: set when robot sends batch_done (2-3 rolls completed), triggers immediate inference
batch_done_received = False
batch_done_lock = threading.Lock()

# Initial step: reference scene (images + robot state + predicates) from first inference.
# Kept across history wipes; cleared only on robot_reset/robot_continue.
# Used when prior belief is missing: compare current view to initial to infer if objects moved.
initial_step = None
pending_initial_snapshot = None  # { 'step_id': int, 'snapshot': dict } set when submitting first inference
initial_step_lock = threading.Lock()

# ------------------------------------------------------------
# Step-through: wait for GUI "Next Step" when step_mode_enabled
# ------------------------------------------------------------
def _monitor_sync_log(msg):
    """Timestamped log for step-sync tracing."""
    t = time.strftime("%H:%M:%S")
    print(f"[MONITOR {t}] {msg}")

def wait_for_next_step(context_msg=""):
    """When step_mode_enabled, pause until user clicks 'Next Step' in GUI. Use context_msg in the log."""
    global step_mode, step_mode_enabled
    with step_mode_lock:
        if not step_mode_enabled:
            return
        step_mode = True
    print(f"[Step mode] Paused: {context_msg}. Click 'Next Step' in GUI to continue.")
    while True:
        with step_mode_lock:
            if not step_mode:
                step_mode = True  # reset for next pause
                return
        time.sleep(0.1)


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
    """Call Gemini API with retries. Prints wall-clock time from call until return."""
    @retry.Retry(predicate=is_retryable, initial=10, multiplier=1.5, deadline=300)
    def inner():
        global client
        return client.models.generate_content(model=model_id, contents=contents, config=gen_config)
    t0 = time.perf_counter()
    out = inner()
    elapsed = time.perf_counter() - t0
    print(f"[API] generate_content returned in {elapsed:.2f}s")
    return out

# ------------------------------------------------------------
# Building prompts 
# ------------------------------------------------------------
def generate_prompt(predicate_pool):

    predicates_str = ", ".join(predicate_pool)
    
    return [f"""
### ROLE
You are a **Vision–Kinematics Predicate Grounding Analyst** for robotic manipulation operating in **INFERENCE MODE under PARTIAL OBSERVABILITY**.

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

Each timestep includes:

#### 1. Robot State
- Joint States
- End-Effector (EE) Pose
- Gripper State

#### 2. Visual Data
- **Raw Front View** — external camera of the scene.
- **Wrist View** — camera on the wrist. **IMPORTANT:** This image shows **only the red tip of the left gripper at the bottom of the image** (and sometimes the red tip of the right gripper). Use it to judge **above(robot_endeffector, X)**:
  - Objects that appear **to the right of the left gripper** in the wrist image → the end effector is **above** them.
  - Objects that appear fully **to the left of the left gripper** → the end effector is **not** above them.

#### 3. Predicate History (WHEN PROVIDED)
- Predicate truth values inferred at previous timesteps

Predicate history represents the system’s **best known world state** unless contradicted.

---

### INITIAL SCENE (when provided)

You may receive an **INITIAL SCENE** block: one reference timestep (images + robot state + predicates) from the **first inference** of this run. It is kept globally and **not** overwritten when the rolling history is cleared (e.g. after recovery).

**Use it to reason about object motion:**
- Compare **current** robot pose and view to the **initial** pose and view.
- **Non-manipulated objects:** If the robot is at the same location and the scene looks similar, an object that was never grasped and is not visible now likely did not move — it may be occluded. Prefer persisting predicates like on(lid, table) unless contradicted.
- **Manipulated objects:** If an object was grasped (grasped(X) was True in history), it may have been moved or dropped. Do NOT persist its location from history — infer on(pepper, table), inside(pepper, pot), etc. from current evidence.
- If the robot moved or the scene clearly changed, use current evidence; the object may have moved.

**When PRIOR BELIEF is also provided:** use it for persistence under occlusion; current evidence overrides when they contradict. On **perception retry**, re-evaluate **grasped(...)** from the current image only.

---

### CORE PRINCIPLE (REFINED)

A predicate is **True** if:
- There is **positive visual evidence**, OR
- The predicate was **True in history** AND there is **no contradicting evidence**

A predicate is **False** only if:
- There is **explicit negative evidence** from vision or kinematics
- OR robot state makes the predicate physically impossible

**Absence of visual evidence ≠ evidence of absence.**

WHEN WORKING WITH LID BEING GRASPED, THE LID IS PART GLASS SO OBJECTS LIKE BELLPEPPER CAN BE SEEN THROUGH. BE CAREFUL IF WHAT IS GRASPED IS THE LID OR THE OBJECT SEEN THROUGH THE GLASS. FOCUS ON WHERE THE TIP OF THE END EFFECTOR IS.
THE END EFFECTOR CAN NOT BE ABOVE MULTIPLE OBJECTS IN THE SCENE AT THE SAME TIME UNLESS THOSE OBJECTS SHARE ON AND INSIDE RELATIONS WITH EACH OTHER.

---

### HISTORY vs MANIPULATION — WHEN TO TRUST HISTORY

**History is a GOOD indicator** for objects that were **NOT manipulated**:
- Example: lid on table, pot on stove, burner. If an object was never grasped and is now occluded, assume it stayed where it was. Persist predicates like `on(lid, table)` from history.
- Non-manipulated + occluded → trust history; object likely still in place.

**History is a POOR indicator** for objects that **WERE manipulated**:
- If `grasped(X)` was True at any prior step, object X was picked up and could have been moved, dropped, or placed elsewhere.
- Example: bell pepper was grasped, then gripper opened. The pepper may now be on the table, in the pot, or dropped outside. Do NOT persist `on(pepper, table)` or `inside(pepper, pot)` from history — use **current visual and kinematic evidence**.
- Manipulated objects can be anywhere; history does NOT tell you where they ended up. Set location predicates (on, above, inside) based on what you see NOW.

**Summary:** 
- Never manipulated + occluded → persist from history.
- Was grasped / manipulated → history is unreliable for location; infer from current evidence.

---
### TEMPORAL CONSISTENCY RULES ( NON-NEGOTIABLE)

1. **Persistence Rule**
   - If a predicate was True at T-1, it remains True at T
     unless explicitly contradicted. Do not blindly copy T-1; ensure consistency (e.g. pot still on stove).
   - **Exception for manipulated objects:** If the object was grasped in history, do NOT persist location predicates (on, above, inside) — infer from current evidence.

2. **Occlusion Rule**
   - If an object is **not manipulated** (never grasped) and is occluded, assume it still exists and persists (e.g. lid on table).
   - If an object **was manipulated** (grasped at some point) and is now occluded or out of view, do NOT assume location — use current evidence only.

3. **Contradiction Priority**
   - Explicit contradictions override history.
   - Examples:
     - Gripper opens → previously grasped object becomes ungrasped

4. **No Hallucinated State Changes**
   - Do NOT infer state changes without causal evidence
     (motion, contact, release, insertion, etc.)

5. **History Can Be Wrong (Avoid Propagating Errors)**
   - If current image or robot state **clearly contradicts** the prior belief, **current evidence wins**.
   - Do not let a single wrong detection in history force the same wrong answer in the current step.

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
If A or B is occluded: defer to history unless contradicted by EE motion or physics.

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
- **Use the Wrist View:** usually only the left gripper is visible (sometimes the tip of the right is visible too). Objects **to the right** of the left gripper in the wrist image → EE is above them; objects **to the left** of the gripper → EE is not above them. BUT THE ENTIRE OBJECT NEEDS TO BE TO THE LEFT IF PART OF IT IS TO THE RIGHT THEN ITS ABOVE.
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

def generate_prompt_old(predicate_pool):

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
- **Raw Front View** — external camera of the scene.
- **Wrist View** — camera on the wrist. **IMPORTANT:** This image shows **only the red tip of the left gripper at the bottom of the image** (and sometimes the red tip of the right gripper). Use it to judge **above(robot_endeffector, X)**:
  - Objects that appear **to the right of the left gripper** in the wrist image → the end effector is **above** them.
  - Objects that appear fully **to the left of the left gripper** → the end effector is **not** above them.

#### 3. Predicate History (WHEN PROVIDED)
- Predicate truth values inferred at previous timesteps

Predicate history represents the system’s **best known world state** unless contradicted.

---

### INITIAL SCENE (when provided)

You may receive an **INITIAL SCENE** block: one reference timestep (images + robot state + predicates) from the **first inference** of this run. It is kept globally and **not** overwritten when the rolling history is cleared (e.g. after recovery).

**Use it to reason about object motion:**
- Compare **current** robot pose and view to the **initial** pose and view.
- If the robot is at the **same location** and the scene looks **similar** (same support surfaces, same layout), an object that was true in the initial scene and is **not visible now** likely did **not** move — it may be occluded or out of frame. Prefer persisting that predicate unless you have **explicit contradicting evidence** (e.g. you see the object elsewhere, or something else is now in that spot).
- If the **robot moved** or the **scene clearly changed** (e.g. different viewpoint, object clearly in a different place), use **current evidence** and the initial reference to decide: the object may have moved, so set False only when you have evidence it is no longer there or is elsewhere.

**When PRIOR BELIEF is also provided:** use it for persistence under occlusion; current evidence overrides when they contradict. On **perception retry**, re-evaluate **grasped(...)** from the current image only.

---

### CORE PRINCIPLE (REFINED)

A predicate is **True** if:
- There is **positive visual evidence**, OR
- The predicate was **True in history** AND there is **no contradicting evidence**

A predicate is **False** only if:
- There is **explicit negative evidence** from vision or kinematics
- OR robot state makes the predicate physically impossible

**Absence of visual evidence ≠ evidence of absence.**

WHEN WORKING WITH LID BEING GRASPED, THE LID IS PART GLASS SO OBJECTS LIKE BELLPEPPER CAN BE SEEN THROUGH> BE CAREFUL IF WHAT IS GRASPED IS THE LID OR THE OBJECT SEEN THROUGH THE GLASS> FOCUS ON WHERE THE TIP OF THE END EFFECTOR IS.
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

5. **History Can Be Wrong (Avoid Propagating Errors)**
   - If current image or robot state **clearly contradicts** the prior belief, **current evidence wins**.
   - Do not let a single wrong detection in history force the same wrong answer in the current step.

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
If A or B is occluded: defer to history unless contradicted by EE motion or physics.

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
- **Use the Wrist View:** usually only the left gripper is visible (sometimes the tip of the right is visible too). Objects **to the right** of the left gripper in the wrist image → EE is above them; objects **to the left** of the gripper → EE is not above them. BUT THE ENTIRE OBJECT NEEDS TO BE TO THE LEFT IF PART OF IT IS TO THE RIGHT THEN ITS ABOVE.
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


def save_inference_images(step_id, front_img, wrist_img, folder):
    """Save front and wrist images used for inference next to the predicate JSON (same step_id)."""
    try:
        if front_img is not None:
            path = os.path.join(folder, f"{step_id}_front.jpg")
            if hasattr(front_img, "save"):
                front_img.save(path, "JPEG", quality=92)
        if wrist_img is not None:
            path = os.path.join(folder, f"{step_id}_wrist.jpg")
            if hasattr(wrist_img, "save"):
                wrist_img.save(path, "JPEG", quality=92)
    except Exception as e:
        print(f"[save_inference_images] Error saving step {step_id}: {e}")


def copy_step_for_initial(step_data):
    """Deep copy one timestep (including PIL images) for storing as initial_step reference."""
    out = {
        "t": step_data.get("t"),
        "joints": step_data.get("joints"),
        "ee": step_data.get("ee"),
        "gripper": step_data.get("gripper"),
        "grasped": step_data.get("grasped", 0),
    }
    if step_data.get("front_img") is not None:
        out["front_img"] = step_data["front_img"].copy()
    else:
        out["front_img"] = None
    if step_data.get("wrist_img") is not None:
        out["wrist_img"] = step_data["wrist_img"].copy()
    else:
        out["wrist_img"] = None
    return out

def _extract_manipulated_objects(last_belief):
    """Extract objects that were grasped (manipulated) from prior belief for HISTORY vs MANIPULATION."""
    if last_belief is None:
        return []
    manipulated = []
    intervals = last_belief.get("intervals", [])
    for iv in intervals:
        preds = iv.get("predicates", {})
        for name, data in preds.items():
            if name.strip().startswith("grasped(") and data.get("is_true") is True:
                manipulated.append(name)
    return list(set(manipulated))


def format_history_prompt(history_buffer, last_belief=None, perception_retry=False, initial_step_ref=None,
                          manipulation_context=None):
    """
    Takes a deque of timestep data and flattens it into Gemini format.
    perception_retry: when True, re-evaluate grasped(...) from current frame only.
    initial_step_ref: optional dict with keys front_img, wrist_img, joints, ee, gripper, predicates.
    manipulation_context: prior belief passed ONLY to identify manipulated objects (grasped); do not persist their locations.
    """
    full_prompt_content = []

    # --- INITIAL SCENE (reference, kept across history wipes) ---
    if initial_step_ref is not None:
        full_prompt_content.append(
            "\n--- INITIAL SCENE (reference — first inference of this run, use to compare if objects moved) ---\n"
        )
        full_prompt_content.append(
            f"Joints: {initial_step_ref.get('joints', '')}\n"
            f"EE Pose: {initial_step_ref.get('ee', '')}\n"
            f"Gripper: {initial_step_ref.get('gripper', '')}\n"
        )
        if initial_step_ref.get("predicates"):
            full_prompt_content.append(
                f"Predicates at initial step:\n{json.dumps(initial_step_ref['predicates'], indent=2)}\n"
            )
        if initial_step_ref.get("front_img"):
            full_prompt_content.append("Initial Front View:")
            full_prompt_content.append(initial_step_ref["front_img"])
        if initial_step_ref.get("wrist_img"):
            full_prompt_content.append("Initial Wrist View:")
            full_prompt_content.append(initial_step_ref["wrist_img"])
        full_prompt_content.append("\n")

    # --- MANIPULATION CONTEXT (when history wiped but we have prior — which objects were grasped) ---
    if manipulation_context is not None:
        manipulated = _extract_manipulated_objects(manipulation_context)
        full_prompt_content.append(
            "\n--- MANIPULATION CONTEXT (history was wiped; use this to identify manipulated objects) ---\n"
        )
        full_prompt_content.append(
            "The following predicates were TRUE at the last inference (objects that were grasped/manipulated):\n"
        )
        if manipulated:
            full_prompt_content.append(f"Manipulated objects: {', '.join(manipulated)}\n")
            full_prompt_content.append(
                "Treat these as MANIPULATED. Do NOT persist their location (on, above, inside) from history. "
                "Infer from current evidence. For non-manipulated objects, use INITIAL SCENE above.\n"
            )
        else:
            full_prompt_content.append("No grasped predicates in prior. All objects may be treated as non-manipulated.\n")
        full_prompt_content.append("\n")

    # --- OBSERVATION CONTEXT ---
    num_frames = len(history_buffer)
    if last_belief is None:
        if initial_step_ref is not None:
            full_prompt_content.append(
                "\n--- OBSERVATION CONTEXT: NO ROLLING PRIOR (e.g. after recovery) — USE INITIAL SCENE ABOVE ---\n"
            )
            full_prompt_content.append(
                "No recent predicate belief. Use the **INITIAL SCENE** above as reference: "
                "compare current robot pose and view to initial. Same location + similar scene → object likely still there if not visible; "
                "robot moved or scene changed → use current evidence to decide.\n"
            )
        else:
            full_prompt_content.append(
                "\n--- OBSERVATION CONTEXT: FIRST FRAME (no prior, no initial scene yet) ---\n"
            )
            if num_frames <= 1:
                full_prompt_content.append("Only one frame provided. Infer from current evidence.\n")
    else:
        full_prompt_content.append(
            "\n--- OBSERVATION CONTEXT: PRIOR BELIEF AVAILABLE ---\n"
        )
        full_prompt_content.append(
            "Use the prior belief below for persistence under occlusion. "
            "Current evidence overrides prior when they contradict. "
        )
        if perception_retry:
            full_prompt_content.append(
                "This is a **perception retry**: re-evaluate all **grasped(...)** from the current image only; "
                "for other predicates, persist from prior unless contradicted.\n"
            )
        else:
            full_prompt_content.append("\n")

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
    global shared_buffer, running, stopped, grasped, initial_step, pending_initial_snapshot, stop_buffer, batch_done_received
    
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
                print(">>> Wiping History Buffer ...") #and Initial Step...")
                with buffer_lock:
                    shared_buffer.clear()
                    stop_buffer = False
                # with initial_step_lock:
                #     initial_step = None
                #     pending_initial_snapshot = None
                continue
            if 'event' in message and message['event'] == 'robot_continue':
                print(">>> Wiping History Buffer ...") #and Initial Step...")
                with buffer_lock:
                    shared_buffer.clear()
                    stop_buffer = False
                # with initial_step_lock:
                #     initial_step = None
                #     pending_initial_snapshot = None
                stopped = False
                continue
            
            if 'event' in message and message['event'] == 'snapshot_sent':
                print("*** RECEIVED SNAPSHOT (processing state)")
                # Fall through: snapshot_sent now includes full state; add to buffer below

            # batch_done / robot_stopped: process as normal state below
            is_batch_done = message.get('event') == 'batch_done'
            is_robot_stopped = message.get('event') == 'robot_stopped'
            
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
            if is_batch_done:
                with batch_done_lock:
                    batch_done_received = True
                _monitor_sync_log("BATCH_DONE received (data receiver) → inference will run")
            if is_robot_stopped:
                _monitor_sync_log("ROBOT_STOPPED received — state-at-stop in buffer for recovery")

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
# --- SEQUENTIAL MODE: No ThreadPoolExecutor; inference runs one-at-a-time ---

# --- WORKER FUNCTION ---
# This runs in the background. We pass 'grasped_state' in so it travels with the request.
def call_gemini_worker(client, model_id, payload, config, step_id, grasped_state, pose):
    try:
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model_id,
            contents=payload,
            config=config
        )
        elapsed = time.perf_counter() - t0
        print(f"[API] Step {step_id} generate_content returned in {elapsed:.2f}s")

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

def trigger_perception_standalone(perception_retry=False, use_initial_only=False, valid_states=None, infer_mode=False):
    """
    Run perception once.
    - perception_retry=True: pass prior belief, re-evaluate grasped(...) from current image only.
    - use_initial_only=True: pass NO rolling prior, only initial_step as reference (grasp-delay case).
    - valid_states: list of valid mode names (e.g. from ltl_monitor.automaton_dict.keys()).
      When provided, add LAST CHANCE RECOVERY instruction so VLM can bias toward a valid state if ambiguous.
    """
    global shared_buffer, running, stopped, last_inferred_predicates, last_inference_time, vis_image_buffer
    global step_counter, initial_step
    
    print("trigger_perception_standalone STARTED")
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
        if current_context_snapshot:
            shared_buffer.clear()  # Consume so we don't reuse same images
    if not current_context_snapshot:
        # Buffer empty; wait briefly for data (e.g. after SNAPSHOT request)
        for _ in range(50):  # ~5s
            time.sleep(0.1)
            with buffer_lock:
                if len(shared_buffer) > 0:
                    current_context_snapshot = list(shared_buffer)
                    shared_buffer.clear()
                    break
        if not current_context_snapshot:
            print("[trigger_perception_standalone] Buffer empty after wait, cannot run inference")
            return None
    last_trigger_time = time.time()

    with initial_step_lock:
        initial_ref = initial_step
    if use_initial_only:
        # Grasp delay: use initial state of the world only (no rolling prior) so model compares current to initial
        prior_for_prompt = None
        perception_retry = False
        manipulation_context = last_inferred_predicates  # pass so model knows which objects were grasped
    else:
        prior_for_prompt = last_inferred_predicates
        manipulation_context = None  # prior already contains full history
    last_step = current_context_snapshot[-1]
    history_payload = format_history_prompt(
        current_context_snapshot,
        last_belief=prior_for_prompt,
        perception_retry=perception_retry,
        initial_step_ref=initial_ref,
        manipulation_context=manipulation_context,
    )
    final_payload = [system_instruction] + history_payload
    if valid_states is not None and len(valid_states) > 0:
        last_chance_text = (
            "\n\n--- LAST CHANCE RECOVERY ---\n"
            "The current predicate assignment did not match any valid state. "
            f"Valid states for this task are: {', '.join(valid_states)}. "
            "If the scene is ambiguous, prefer predicate assignments that match one of these states so the task can continue. "
            "Only output a valid-state-matching assignment if consistent with the images; do not hallucinate.\n"
        )
        final_payload = final_payload + [last_chance_text]

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
        with initial_step_lock:
            if initial_step is None:
                initial_step = {
                    **copy_step_for_initial(last_step),
                    "predicates": copy.deepcopy(json_output),
                }
                print("Initial step saved (reference scene).")
        last_inferred_predicates = json_output
        last_inference_time = time.time()
        
        output_file = os.path.join(OUTPUT_FOLDER, f"{step_counter}_true_predicates.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=4)
        save_inference_images(step_counter, last_step.get("front_img"), last_step.get("wrist_img"), OUTPUT_FOLDER)
        step_counter += 1
        print("Successfully parsed JSON.")            
        
        ## if we get a response 
        pred_vector = json_to_predicate_vector(
            json_output,
            predicate_to_idx,
            PRED_DIM
        )
        
        
        pred_vector = np.append(pred_vector, current_grasped_state)
        print("pred_vector after appending: ",pred_vector)
        # update the predicates
        ltl_monitor.predicates = pred_vector
        # run a step
        if not infer_mode:
            print(f"Not infer mode, stepping through")
            status_monitor = ltl_monitor.monitor_step()
            status = status_monitor.status
        else:
            print(f"infer mode, stepping through")
            inferred_mode = ltl_monitor.infer_mode_from_predicates(pred_vector)
            if inferred_mode is not None:
                status = "running"
                ## set it
                ltl_monitor.curr_mode = inferred_mode
            else:
                status = "violation"
        
        
        print("LTL Monitor Status: ", status)
        
        if status == "violation":
            print("!!! LTL VIOLATION — SENDING STOP TO ROBOT !!!")
            print("trigger_perception_standalone ENDED")
        elif status == "done":
            print("!!! LTL MONITOR: TASK DONE DETECTED !!!")
            print("trigger_perception_standalone ENDED")
        return  status   
  
def set_exec_visual_state(state: ExecVisualState):
    global exec_visual_state
    with exec_visual_lock:
        exec_visual_state = state
          
def llm_monitor_loop_sim(ltl_monitor, cmd_pub_socket):
    global shared_buffer, running, stopped, last_inferred_predicates, last_inference_time, vis_image_buffer
    global client, initial_step, pending_initial_snapshot, batch_done_received
    global step_counter, exec_visual_state, stop_buffer
    print("LLM Monitor: Started (Sequential Mode — one inference at a time).")

    last_recovery = None

    # --- 1. SETUP MULTI-KEY ROTATION ---
    # Load all keys
    if paid:
        keys = [
            os.environ.get("GEMINI_API_KEY_CLOUD")
        ]
    else:
        keys = [
        os.environ.get("GEMINI_API_KEY_HASSAN"),
        os.environ.get("GEMINI_API_KEY_NOUR"),
        os.environ.get("GEMINI_API_KEY_LEEN"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_GMAIL"),
    ]
    # Filter out None keys if environment vars are missing
    keys = [k for k in keys if k]
    
    # Create a client for each key (round-robin for sequential calls)
    clients = [genai.Client(api_key=k) for k in keys]
    client_idx = 0
    print(f"Loaded {len(clients)} API clients for rotation.")

    # --- 2. STANDARD SETUP ---
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
    
    enforcer = LogicEnforcer(predicate_to_idx, "/home/olagh48652/task_monitor/test_rules/rules.yaml")
    
    grasped_indices = [i for i, p in enumerate(predicate_pool) if p.strip().startswith("grasped(")]
    last_inferred_predicates = None
    last_trigger_time = 0
    # Step-sync: inference driven by batch_done only (no timer trigger)

    keypoints = KeypointStore(reset_pose=np.zeros(7))
    violation_retry_count = 0
    grasp_recovery = 0
    stop_buffer = False
    buffer_ever_had_data = False  # True after first time we received images (needed to allow SNAPSHOT request)
    # Stuck recovery: if robot stays in same mode for this many seconds, go to reset and restart from there
    STUCK_THRESHOLD_SEC = 60.0* 2
    stuck_since = None  # time.time() when we first saw current mode with no transition; None if just transitioned
    while running:
        if stopped:
            time.sleep(1)
            continue
        
        set_exec_visual_state(ExecVisualState.RUNNING)
        
        current_time = time.time()
        
        # ==========================================================
        # PHASE A: SUBMISSION — Step-sync ONLY: trigger on batch_done
        # (15s timer removed: inference driven by robot handshake only)
        # ==========================================================
        with batch_done_lock:
            batch_ready = batch_done_received
        with buffer_lock:
            has_data = len(shared_buffer) > 0
        
        if not (batch_ready and has_data):
            time.sleep(0.2)  # Wait for batch_done, no busy spin
            continue

        with batch_done_lock:
            batch_done_received = False
        _monitor_sync_log("batch_done consumed → running inference (step-sync)")

        # 1. Get Snapshot (consume buffer so we don't send same images twice)
        current_context_snapshot = []
        with buffer_lock:
            if len(shared_buffer) > 0:
                current_context_snapshot = list(shared_buffer)
                shared_buffer.clear()  # Remove used images so next inference gets new ones
                buffer_ever_had_data = True
            else:
                pass  # Buffer empty; handle below

        # 2. If buffer was empty: wait for first images, or request SNAPSHOT (only after we've had data once)
        if not current_context_snapshot:
            if not buffer_ever_had_data:
                # First time: don't request snapshot, just wait for images to arrive
                time.sleep(0.2)
                continue
            # We've had data before; request snapshot and wait for it
            _monitor_sync_log("[Buffer empty] Requesting SNAPSHOT...")
            cmd_pub_socket.send_pyobj({"command": "SNAPSHOT"})
            wait_start = time.time()
            SNAPSHOT_TIMEOUT = 15.0
            while time.time() - wait_start < SNAPSHOT_TIMEOUT:
                with buffer_lock:
                    if len(shared_buffer) > 0:
                        current_context_snapshot = list(shared_buffer)
                        shared_buffer.clear()
                        _monitor_sync_log("[Buffer] Snapshot received, proceeding with inference")
                        break
                time.sleep(0.1)
            else:
                _monitor_sync_log("[Buffer] Timeout waiting for snapshot, skipping")
                continue

        if current_context_snapshot:
            last_trigger_time = current_time  # Reset timer

            # Send PROCEED immediately so robot can start next batch while inference
            _monitor_sync_log("PROCEED sent (robot can move) → running inference in background")
            cmd_pub_socket.send_pyobj({"command": "PROCEED"})

            # Prepare Prompt (Main thread does this quickly)
            last_step = current_context_snapshot[-1]
            with initial_step_lock:
                initial_ref = initial_step  # read under lock; no copy needed
            history_payload = format_history_prompt(
                current_context_snapshot,
                last_belief=last_inferred_predicates,
                perception_retry=False,
                initial_step_ref=initial_ref,
            )
            final_payload = [system_instruction] + history_payload

            # If this is the first inference, save snapshot so we can set initial_step when result arrives
            with initial_step_lock:
                if last_inferred_predicates is None and initial_step is None and pending_initial_snapshot is None:
                    pending_initial_snapshot = {
                        "step_id": step_counter,
                        "snapshot": copy_step_for_initial(last_step),
                    }

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

            _monitor_sync_log("inference START step %d (buffer_frames=%d)" % (step_counter, len(current_context_snapshot)))
            save_inference_images(step_counter, last_step.get("front_img"), last_step.get("wrist_img"), OUTPUT_FOLDER)

            # SEQUENTIAL: Call API directly and wait for result (no parallel submission)
            result = call_gemini_worker(
                active_client,
                MODEL_ID,
                final_payload,
                gen_config,
                step_counter,
                current_grasped_state,
                current_pose,
            )
            step_counter += 1

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
                        # If this result is from the first inference, set initial_step (reference scene for later wipes)
                        with initial_step_lock:
                            if pending_initial_snapshot is not None and r_step_id == pending_initial_snapshot["step_id"]:
                                initial_step = {
                                    **pending_initial_snapshot["snapshot"],
                                    "predicates": copy.deepcopy(r_json),
                                }
                                pending_initial_snapshot = None
                                print("Initial step saved (reference scene for post-recovery inference).")

                        last_inferred_predicates = r_json
                        last_inference_time = time.time()
                        
                        # Save logs (Keep your existing logic)
                        output_file = os.path.join(OUTPUT_FOLDER, f"{r_step_id}_true_predicates.json")
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(r_json, f, indent=4)

                        pred_vector_unenforced = json_to_predicate_vector(r_json, predicate_to_idx, PRED_DIM)
                        
                        print(f"BEFORE {pred_vector_unenforced}")
                        pred_vector = enforcer.apply_rules(pred_vector_unenforced)
                        
                        print(f"AFTER {pred_vector}")
                        
                        r_json = None
                        
                        # Grasped scalar: use proprioceptive r_grasped, but override to 1 if VLM says any grasped(X)=true
                        # (proprioceptive can be 0 due to encoding/threshold/timing while VLM sees grasp)
                        grasped_for_vector = r_grasped
                        if any(pred_vector[i] == 1 for i in grasped_indices):
                            grasped_for_vector = 1
                        if grasped_for_vector != r_grasped:
                            print(f"[grasped] Override: proprioceptive={r_grasped} -> {grasped_for_vector} (VLM has grasped pred)")
                        pred_vector = np.append(pred_vector, grasped_for_vector) 

                        ltl_monitor.predicates = pred_vector
                        mode_before = ltl_monitor.curr_mode  # capture before monitor updates it
                        result = ltl_monitor.monitor_step()
                        print("result ",result)
                        print(f"Step {r_step_id} LTL Status: {result.status}")

                        if result.status == "running":
                            grasp_recovery = 0
                            # Step-sync: tell robot inference ok, can continue next batch
                            _monitor_sync_log("inference OK step %d → sending PROCEED" % r_step_id)
                            cmd_pub_socket.send_pyobj({"command": "PROCEED"})
                            # Stuck detection: reset timer when we transitioned (mode changed).
                            # Note: monitor updates curr_mode before returning, so result.next_mode==result.curr_mode
                            # even right after a transition; we must compare to mode_before.
                            if result.curr_mode != mode_before or result.next_mode != result.curr_mode:
                                stuck_since = None  # we transitioned, reset stuck timer
                            else:
                                now = time.time()
                                if stuck_since is None:
                                    stuck_since = now
                                elif now - stuck_since >= STUCK_THRESHOLD_SEC:
                                    print(f"!!! STUCK in mode {result.curr_mode} for {now - stuck_since:.0f}s — recovery: reset and restart !!!")
                                    stuck_since = None
                                    set_exec_visual_state(ExecVisualState.RECOVERING)
                                    
                                    cmd_pub_socket.send_pyobj({"command": "STOP", "reason": "stuck_recovery"})
                                    stopped = True
                                    
                                    wait_for_next_step("Threashold hit robot stuck to mode — send CONTINUE?")
                                    
                                    cmd_pub_socket.send_pyobj({"command": "RESET_POSE"})
                                    
                                    wait_for_next_step("take snapshot — send CONTINUE?")
                                    cmd_pub_socket.send_pyobj({"command": "SNAPSHOT"})
                                    # Wait for snapshot to arrive
                                    wait_start = time.time()
                                    while time.time() - wait_start < 15.0:
                                        with buffer_lock:
                                            if len(shared_buffer) > 0:
                                                break
                                        time.sleep(0.1)
                                    new_stat = trigger_perception_standalone(infer_mode=True)
                                    inferred = ltl_monitor.infer_mode_from_predicates(ltl_monitor.predicates) if ltl_monitor.predicates is not None else None
                                    if inferred is not None:
                                        ltl_monitor.curr_mode = inferred
                                        print(f"[Stuck recovery] Mode set to {inferred}, continuing")
                                        stopped = False
                                        cmd_pub_socket.send_pyobj({"command": "CONTINUE"})
                                        set_exec_visual_state(ExecVisualState.CONTINUING)
                                    else:
                                        print("[Stuck recovery] No valid state after reset, staying stopped")

                        if result.status == "violation":
                            stuck_since = None
                            print("!!! VIOLATION DETECTED !!!")
                            
                            # Gripper closed but no grasped predicate true: either failed grasp or inference delay (grasped often true only after lift)
                            if r_grasped and not any(pred_vector[i] == 1 for i in grasped_indices):
                                print("any grasped predicate true:", any(pred_vector[i] == 1 for i in grasped_indices))
                                print(f"r_grasped={r_grasped} but no grasped pred true — delay or failed grasp")
                                if grasp_recovery:
                                    print("Second time: treat as real violation (robot likely failed to grasp)")
                                    violation_retry_count += 1
                                else:
                                    grasp_recovery = 1
                                    # Buffer is empty (consumed by main inference); request fresh snapshot for retry
                                    _monitor_sync_log("Grasp retry: requesting SNAPSHOT for fresh obs")
                                    cmd_pub_socket.send_pyobj({"command": "SNAPSHOT"})
                                    wait_start = time.time()
                                    while time.time() - wait_start < 15.0:
                                        with buffer_lock:
                                            if len(shared_buffer) > 0:
                                                break
                                        time.sleep(0.1)
                                    else:
                                        _monitor_sync_log("Grasp retry: timeout waiting for SNAPSHOT")
                                    print("Retrying perception with initial scene only (no rolling prior)")
                                    new_stat = trigger_perception_standalone(use_initial_only=True)
                                    
                                    print(f"new_stat: {new_stat}")
                                    if new_stat == "running":
                                        print(f"new_stat: running")
                                        grasp_recovery = 0
                                        print("publishing continue")
                                        violation_retry_count = 0
                                        last_recovery = None
                                        cmd_pub_socket.send_pyobj({"command": "PROCEED"})
                                    continue
                                    
                            
                            set_exec_visual_state(ExecVisualState.VIOLATION)
                            _monitor_sync_log("VIOLATION step %d → sending STOP" % r_step_id)
                            stop_payload = {"command": "STOP", "reason": "LTL violation"}
                            cmd_pub_socket.send_pyobj(stop_payload)
                            stopped = True

                            # Recovery loop: keep deciding and executing until we continue or abort.
                            # After RETRY_PERCEPTION we re-infer from updated predicates and re-decide (e.g. RECOVER_TO_MODE(ag)).
                            current_mode_for_recovery = result.curr_mode
                            recovery_done = False
                            while not recovery_done:
                                inferred_mode = ltl_monitor.infer_mode_from_predicates(ltl_monitor.predicates)
                                print(f"ltl_monitor.predicates: {ltl_monitor.predicates}")
                                print(f"[recovery] inferred_mode={inferred_mode} current_mode={current_mode_for_recovery}")
                                action, target = decide_recovery(
                                    inferred_mode=inferred_mode,
                                    current_mode=current_mode_for_recovery,
                                    retry_count=violation_retry_count,
                                    keypoint_store=keypoints,
                                    last_recovery=last_recovery
                                )
                                violation_retry_count += 1
                                print(f"[recovery] action={action} target={target}")

                                if action == RecoveryAction.RECOVER_TO_MODE:
                                    set_exec_visual_state(ExecVisualState.RECOVERING)
                            
                                    ## for the vlm to infer it the robot should be in that mdoe so no need to move just set it.
                                    print(f"target: {target}")
                                    ltl_monitor.curr_mode = target
                                    wait_for_next_step("Recovering to mode — send CONTINUE?")
                                    print("CONTINUE...")
                                    cmd_pub_socket.send_pyobj({"command": "CONTINUE"})
                                    grasp_recovery = 0
                                    stopped = False
                                    violation_retry_count = 0
                                    last_recovery = None
                                    recovery_done = True

                                elif action == RecoveryAction.RECOVER_GET_STATE:
                                    set_exec_visual_state(ExecVisualState.RECOVERING)
                                    print(f"Recovering to mode with reset position")
                                    cmd_pub_socket.send_pyobj({"command": "RESET_POSE"})
                                    # time.sleep(2)
                                    wait_for_next_step("Recovering to reset pose — run perception?")
                                    print("RUNNING PERCEPTION")
                                    
                                    ## get current snapshot
                                    cmd_pub_socket.send_pyobj({"command": "SNAPSHOT"})
                                    # Wait for snapshot to arrive before perception (ensures fresh obs)
                                    wait_start = time.time()
                                    while time.time() - wait_start < 15.0:
                                        with buffer_lock:
                                            if len(shared_buffer) > 0:
                                                break
                                        time.sleep(0.1)
                                    wait_for_next_step("Recovering to reset pose — run perception?")
                                    new_stat = trigger_perception_standalone(infer_mode=True)
                                    print(f"new_stat: {new_stat}")
                                    if new_stat == "running":
                                        print("new_stat: running")
                                        violation_retry_count = 0
                                        last_recovery = None
                                        wait_for_next_step("Recovering to reset pose — send CONTINUE?")
                                        print("CONTINUE...")
                                        cmd_pub_socket.send_pyobj({"command": "CONTINUE"})
                                        grasp_recovery = 0
                                        stopped = False
                                        violation_retry_count = 0
                                        last_recovery = None
                                        recovery_done = True
                                            
                                    else :
                                        print("!!! No state satisfied after last-chance recovery — ABORT (task cannot be done) !!!")
                                        set_exec_visual_state(ExecVisualState.DONE)
                                        cmd_pub_socket.send_pyobj({
                                            "command": "ABORT",
                                            "reason": "No valid state after recovery; task cannot be completed",
                                        })
                                        stopped = True
                                        last_recovery = action
                                        recovery_done = True
                                    
                                elif action == RecoveryAction.RETRY_PERCEPTION:
                                    set_exec_visual_state(ExecVisualState.RECOVERING)
                                    # Buffer empty (robot stopped); request fresh snapshot for retry
                                    _monitor_sync_log("RETRY_PERCEPTION: requesting SNAPSHOT for fresh obs")
                                    cmd_pub_socket.send_pyobj({"command": "SNAPSHOT"})
                                    wait_start = time.time()
                                    while time.time() - wait_start < 15.0:
                                        with buffer_lock:
                                            if len(shared_buffer) > 0:
                                                break
                                        time.sleep(0.1)
                                    else:
                                        _monitor_sync_log("RETRY_PERCEPTION: timeout waiting for SNAPSHOT")
                                    print("Retrying perception (re-evaluating grasp from image, persisting other predicates)")
                                    new_stat = trigger_perception_standalone(perception_retry=True)
                                    print(f"new_stat: {new_stat}")
                                    if new_stat == "running":
                                        print(f"new_stat: running")
                                        grasp_recovery = 0
                                        stopped = False
                                        cmd_pub_socket.send_pyobj({"command": "CONTINUE"})
                                        set_exec_visual_state(ExecVisualState.CONTINUING)
                                        violation_retry_count = 0
                                        last_recovery = None
                                        recovery_done = True
                                    else:
                                        last_recovery = action
                                        current_mode_for_recovery = ltl_monitor.curr_mode
                                        # Loop again: re-infer from updated predicates and re-decide (e.g. RECOVER_TO_MODE(ag))

                                elif action == RecoveryAction.ABORT:
                                    print("!!! ABORT: still no state after last chance — task cannot be completed !!!")
                                    set_exec_visual_state(ExecVisualState.DONE)
                                    cmd_pub_socket.send_pyobj({
                                        "command": "ABORT",
                                        "reason": "Recovery exhausted; no valid state (task cannot be completed)",
                                    })
                                    stopped = True
                                    recovery_done = True
                                    break

                        ## update keypoint after a successful transition (curr_mode changed from mode_before)
                        if result.status == "running" and result.curr_mode != mode_before:
                            keypoints.save(
                                mode=result.curr_mode,
                                pose=r_pose,
                                grasped=r_grasped
                            )
    
                        elif result.status == "done":
                            stuck_since = None
                            print("!!! TASK DONE !!!")
                            done_payload = {"command": "TASK_DONE", "reason": "Success"}
                            set_exec_visual_state(ExecVisualState.DONE)
                            cmd_pub_socket.send_pyobj(done_payload)
                            stopped = True
                            running = False
                            break

                    except Exception as e:
                        print(f"Error parsing/updating LTL for step {r_step_id}: {e}")
        
        # Small sleep to prevent CPU spinning while waiting for IO or timers
        time.sleep(0.05)
                   
import threading
import tkinter as tk

def exec_status_light():
    root = tk.Tk()
    root.title("Execution Status")
    root.geometry("180x100")
    canvas = tk.Canvas(root, width=180, height=100)
    canvas.pack()

    circle = canvas.create_oval(40, 20, 80, 60, fill="green")
    text = canvas.create_text(110, 40, text="RUNNING", font=("Arial", 10, "bold"))

    def update():
        

        with exec_visual_lock:
            state = exec_visual_state

        color_map = {
            ExecVisualState.RUNNING: "green",
            ExecVisualState.VIOLATION: "red",
            ExecVisualState.RECOVERING: "orange",
            ExecVisualState.CONTINUING: "blue",
            ExecVisualState.ABORT: "yellow",
            ExecVisualState.DONE: "white",
        }

        canvas.itemconfig(circle, fill=color_map[state])
        canvas.itemconfig(text, text=state.value)
        root.after(100, update)

    update()
    root.mainloop()
    
def gui_backdoor_buttons(ltl_monitor):
    global system_instruction, client
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

    # Step-through mode: pause at recovery points until "Next Step" is clicked
    with step_mode_lock:
        step_mode_var = tk.BooleanVar(value=step_mode_enabled)

    def toggle_step_mode():
        global step_mode_enabled
        with step_mode_lock:
            step_mode_enabled = step_mode_var.get()
        status_label.config(text=f"Step mode: {'ON' if step_mode_enabled else 'OFF'}")

    def next_step():
        global step_mode
        with step_mode_lock:
            step_mode = False
        status_label.config(text="Next Step clicked — continuing")

    tk.Label(root, text="Step-through (pause at recovery until Next):").pack(padx=10, pady=(10, 0))
    tk.Checkbutton(root, text="Step mode", variable=step_mode_var, command=toggle_step_mode).pack(padx=10, pady=2)
    tk.Button(root, text="Next Step", command=next_step).pack(padx=10, pady=5)

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
        global step_counter, initial_step
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

        prior_for_prompt = last_inferred_predicates
        with initial_step_lock:
            initial_ref = initial_step
        last_step = current_context_snapshot[-1]
        history_payload = format_history_prompt(
            current_context_snapshot,
            last_belief=prior_for_prompt,
            perception_retry=False,
            initial_step_ref=initial_ref,
        )
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
            with initial_step_lock:
                if initial_step is None:
                    initial_step = {
                        **copy_step_for_initial(last_step),
                        "predicates": copy.deepcopy(json_output),
                    }
                    print("Initial step saved (reference scene).")
            last_inferred_predicates = json_output
            last_inference_time = time.time()
            
            output_file = os.path.join(OUTPUT_FOLDER, f"{step_counter}_true_predicates.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=4)
            save_inference_images(step_counter, last_step.get("front_img"), last_step.get("wrist_img"), OUTPUT_FOLDER)
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
            if status.status == "violation":
                print("!!! LTL VIOLATION — SENDING STOP TO ROBOT !!!")
            elif status.status == "done":
                print("!!! LTL MONITOR: TASK DONE DETECTED !!!")

    # Buttons
    tk.Button(root, text="Get Current Predicates", command=get_preds).pack(padx=5, pady=5, fill='x')
    tk.Button(root, text="Get Current Mode", command=get_mode).pack(padx=5, pady=5, fill='x')
    tk.Button(root, text="Set Predicates (no mode update)", command=set_preds_no_update).pack(padx=5, pady=5, fill='x')
    tk.Button(root, text="Set Predicates (update mode)", command=set_preds_update).pack(padx=5, pady=5, fill='x')
    tk.Button(root, text="Trigger Perception", command=trigger_perception).pack(padx=5, pady=5, fill='x')

    # Keep step_mode_var in sync with global on startup
    def sync_step_mode():
        with step_mode_lock:
            step_mode_var.set(step_mode_enabled)
    root.after(100, sync_step_mode)

    root.mainloop()
   
# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    global client, GEMINI_API_KEY, MODEL_ID, paid
    
    # MODEL_ID = "gemini-2.5-pro"
    MODEL_ID = "gemini-2.5-flash"
    # MODEL_ID = "gemini-robotics-er-1.5-preview"
    
    paid = True
    
    if paid: 
        GEMINI_API_KEY = os.environ["GEMINI_API_KEY_CLOUD"]
        api_keys_used.append("GEMINI_API_KEY_CLOUD")
    else:
        GEMINI_API_KEY = os.environ["GEMINI_API_KEY_LEEN"]
        api_keys_used.append("GEMINI_API_KEY_LEEN")
    
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # --- COMMAND PUBLISHER (Monitor -> Robot) ---
    cmd_context = zmq.Context()
    cmd_pub_socket = cmd_context.socket(zmq.PUB)

    # IMPORTANT: bind here, robot connects
    cmd_pub_socket.bind("tcp://*:5546")

    # Gemini Robotics ER 1.5 Preview
    
    # Give ZMQ time to establish the connection
    time.sleep(0.5)
    
    # 1. Start Receiver in Background
    t_recv = threading.Thread(target=data_receiver_thread, daemon=True)
    t_recv.start()

    ltl_monitor = LTLMonitor()
    
    # Start the GUI backdoor in a separate thread
    gui_thread = threading.Thread(target=gui_backdoor_buttons, args=(ltl_monitor,), daemon=True)
    gui_thread.start()
    
    exec_status_light_ = threading.Thread(target=exec_status_light, daemon=True)
    exec_status_light_.start()
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

# rack 

# {'above(bowl, dish_rack)': 0,
#  'above(bowl, table)': 1,
#  'above(cup, dish_rack)': 2,
#  'above(cup, table)': 3,
#  'above(robot_endeffector, bowl)': 4,
#  'above(robot_endeffector, cup)': 5,
#  'above(robot_endeffector, dish_rack)': 6,
#  'above(robot_endeffector, table)': 7,
#  'grasped(bowl)': 8,
#  'grasped(cup)': 9,
#  'inside(bowl, dish_rack)': 10,
#  'inside(cup, dish_rack)': 11,
#  'on(bowl, table)': 12,
#  'on(cup, table)': 13,
#  'on(dish_rack, table)': 14}