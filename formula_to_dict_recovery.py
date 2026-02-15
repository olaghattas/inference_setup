#!/usr/bin/env python3
"""Example publisher that sends FloatVector messages."""

import time
import math

import spot
spot.setup()
import matplotlib.pyplot as plt
import numpy as np
import random
import re

from dataclasses import dataclass
from typing import Optional

@dataclass
class MonitorResult:
    status: str                 # "running" | "violation" | "done"
    curr_mode: str
    next_mode: Optional[str]
    sensor_vec: str


COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"
    
class LTLMonitor:
    def __init__(self):

        filename = "/home/olagh48652/task_monitor/task_pot/task_pot_formula.txt"    
        formula = self.load_formula_from_file(filename)
        # print(formula)
        self.formula = formula
        aut = spot.translate(spot.formula(formula), "ba", "small", "high")

        # extract predicate letters
        self.APs = sorted([str(p) for p in aut.ap()])
        self.env_APs = [ap for ap in self.APs if len(ap)==1]
        self.robot_APs = [ap for ap in self.APs if len(ap)==2]

        print(f"env_APs: {self.env_APs}")
        print(f"robot_APs: {self.robot_APs}")
        
        # Build transition dictionary
        self.node_to_mode = self.get_node_to_mode(aut)
        self.automaton_dict = self.build_automaton_dict(aut)

        # start mode
        self.init_mode = self.node_to_mode[aut.get_init_state_number()]
        self.curr_mode = self.init_mode

        self.predicates = None  # latest measurement
        self.prev_mode = None
        self.prev_preds = None
        
        self.done = False
        
        ## this doesnt find the done sata fix or pass the done state when savign the formula
        self.done_ap = self.find_self_only(formula)
        # print("self.done_ap: ", self.done_ap )
        self.done_ap = ["al"]
        # print("self.done_ap: ", self.done_ap )
            
    def keep_single_letter_preds(self, s):
        tokens = re.findall(r'!?[A-Za-z]+', s)
        out = []
        for t in tokens:
            neg = t.startswith('!')
            name = t[1:] if neg else t

            if len(name) == 1:  # keep ONLY single-letter predicates
                out.append(t)

        return " & ".join(out)

    def get_node_to_mode(self, a):
        node_to_mode = {}
        for i in range(a.num_edges()): 
            edge = a.edge_storage(i+1) # 1 indexed
            destination_node = edge.dst
            
            bdd = spot.bdd_format_formula(a.get_dict(), edge.cond)
            
            for p in self.robot_APs: # only one robot_ap will be true, i.e. one and only one mode at a time
                if '!'+p in bdd: # false assignment to p
                    pass
                else: # true assignment to p. If p is not in bdd, we assume p has been and stays true
                    node_to_mode[destination_node] = p

        node_to_mode[1] = "aa" # since node 1 is a placeholder, we add it to the starting mode
        return node_to_mode
    
    # PARSE DEFINITIONS
    def _get_constraints_from_formula(self):
        """
        Parses the LTL formula to find exact definitions like:
        G(ag <-> (!b & !c ...))
        Returns a dict: {'ag': {'b': '0', 'c': '0'}, ...}
        """
        constraints = {}
        # Regex to find G( mode <-> ( definitions ) )
        # Matches: G(aa <-> (...))
        pattern = r"G\s*\(\s*([a-z]{2})\s*<->\s*\(([^)]+)\)\s*\)"
        matches = re.findall(pattern, self.formula)

        for mode, definition in matches:
            constraints[mode] = {}
            # Split by & to get individual predicates like '!a', 'b'
            preds = re.findall(r'!?[a-z]', definition)
            for p in preds:
                if p.startswith('!'):
                    constraints[mode][p[1:]] = '0' # False
                else:
                    constraints[mode][p] = '1'     # True
        return constraints
    
    def build_automaton_dict(self, a):
        # 1. Standard Spot Extraction
        raw_dict = {} 
        for mode in self.robot_APs:
            raw_dict[mode] = {}
            
        for i in range(a.num_edges()): 
            edge = a.edge_storage(i+1) 
            source_node = edge.src

            bdd = spot.bdd_format_formula(a.get_dict(), edge.cond)
            bdd_env_preds =  self.keep_single_letter_preds(bdd) 
            
            for p in self.robot_APs: 
                if '!'+p in bdd: 
                    pass
                else: 
                    sensor_vec = ''
                    for q in self.env_APs:
                        if '!'+q in bdd_env_preds:
                            sensor_vec+='0'
                        elif q in bdd_env_preds:
                            sensor_vec+='1'
                        else:
                            sensor_vec+='-'                            
                    
                    src_mode = self.node_to_mode.get(source_node)
                    if src_mode:
                        raw_dict[src_mode][sensor_vec] = p

        # 2. RELAXATION LOGIC (Fixing the "Don't Cares")
        # Parse the formula to get the "Source of Truth"
        ltl_constraints = self._get_constraints_from_formula()
        
        # Map variable names to vector indices for fast lookups
        # e.g., {'a': 0, 'b': 1, ...}
        var_to_idx = {name: i for i, name in enumerate(self.env_APs)}

        final_dict = {}

        for state, transitions in raw_dict.items():
            final_dict[state] = {}
            
            for key, target_mode in transitions.items():
                # If we don't have an explicit LTL definition for the target, keep original key
                if target_mode not in ltl_constraints:
                    final_dict[state][key] = target_mode
                    continue
                
                # Get the strict requirements for the target mode
                reqs = ltl_constraints[target_mode]
                
                # Convert immutable string key to mutable list
                key_chars = list(key)
                
                # Iterate through all environment variables
                for var_name, idx in var_to_idx.items():
                    if var_name in reqs:
                        # If LTL says it MUST be X, force it to X
                        # This fixes cases where Spot might have picked a valid path but not the only path
                        key_chars[idx] = reqs[var_name]
                    else:
                        # If LTL does NOT mention this variable, it MUST be a Don't Care
                        # This overrides Spot forcing a 0 or 1 for disambiguation
                        key_chars[idx] = '-'
                
                new_key = "".join(key_chars)
                final_dict[state][new_key] = target_mode

        return final_dict
    
    def find_self_only(self, formula_str):
        results = []

        # extract each transition clause of the form:  G( <pred> -> ( ... ) )
        transition_clauses = re.findall(r"G\(\s*([a-z]+)\s*->\s*\(([^)]*)\)", formula_str)

        for left, rhs in transition_clauses:
            # find all items like Xaa, Xbb, Xcc inside the RHS
            next_items = re.findall(r"X([a-z]+)", rhs)

            # self-only condition: only Xleft appears
            if len(next_items) == 1 and next_items[0] == left:
                results.append(left)

        return results

    def _match_pattern(self, vec, pattern):
        for v, p in zip(vec, pattern):
            if p == '-':
                continue
            if v != p:
                return False
        return True
    
    def monitor_step(self):
        if self.predicates is None:
            print("self.predicates  is None")
            return MonitorResult("None", self.curr_mode, self.curr_mode, "")

        sensor_vec = "".join(str(int(v)) for v in self.predicates)
        transitions = self.automaton_dict[self.curr_mode]

        # Logging when mode changes
        if self.prev_mode != self.curr_mode:
            print(f"{COLOR_GREEN}curr_mode: {self.curr_mode}{COLOR_RESET}")
            print(f"{COLOR_GREEN}   Allowed: {list(transitions.keys())}{COLOR_RESET}")
            self.prev_mode = self.curr_mode
        
        print(f"[monitor step]  predicates: {self.predicates}")
        print(f"[monitor step]  prev_preds: {self.prev_preds}")
        print(f"[monitor step] predicates (indices=1): {[i for i, v in enumerate(self.predicates) if v==1]}")
        if self.prev_preds is not None:
            print(f"[monitor step] prev_preds (indices=1): {[i for i, v in enumerate(self.prev_preds) if v==1]}")
        
        # Only react when predicates actually changed
        if not np.array_equal(self.prev_preds, self.predicates):
            matched = False
            next_mode = self.curr_mode

            for pattern, target in transitions.items():
                if self._match_pattern(sensor_vec, pattern):
                    matched = True
                    next_mode = target
                    break

            if not matched:
                print(f"{COLOR_RED}*** Illegal transition: mode={self.curr_mode}, sensor={sensor_vec}{COLOR_RESET}")
                print(f"{COLOR_RED}   Allowed patterns: {list(transitions.keys())}{COLOR_RESET}")
                # self.prev_preds = self.predicates
                # return "violation"
                print("violation returned")
                return MonitorResult(
                    status="violation",
                    curr_mode=self.curr_mode,
                    next_mode=None,
                    sensor_vec=sensor_vec
                )

            if next_mode != self.curr_mode:
                print(f"{COLOR_GREEN}Transition: {self.curr_mode} --[{sensor_vec}]--> {next_mode}{COLOR_RESET}")

            self.curr_mode = next_mode

            if self.curr_mode in self.done_ap:
                self.done = True
                print(f"{COLOR_BLUE}Task done: {self.curr_mode} is in {self.done_ap}{COLOR_RESET}")
                # return "done"
                print("done returned")
                return MonitorResult("done", self.curr_mode, self.curr_mode, sensor_vec)


            self.prev_preds = self.predicates
            # return "running"
            print("running returned")
            return MonitorResult("running", self.curr_mode, next_mode, sensor_vec)
        
        else:
            self.prev_preds = self.predicates
            return MonitorResult(
                status="running",
                curr_mode=self.curr_mode,
                next_mode=self.curr_mode,
                sensor_vec=sensor_vec
            )

    def infer_mode_from_predicates(self, predicate_vector):
        """
        Infer which mode the current predicates CLAIM we are in.
        Does NOT advance the automaton.
        """
        sensor_vec = "".join(str(int(v)) for v in predicate_vector)

        for mode, transitions in self.automaton_dict.items():
            for pattern in transitions.keys():
                if self._match_pattern(sensor_vec, pattern):
                    return mode
        return None
       
            
    def load_formula_from_file(self, filename="formula.txt"):
        with open(filename, "r") as f:
            return f.read().strip()
        

    def get_mode_from_predicates(self, predicate_vector):
        """
        Given a predicate vector (list/array of 0/1 values), 
        return the current mode according to the automaton_dict.
        Returns None if no mode matches.
        """
        print("[Backdoor] Settiget_mode_from_predicates to:", predicate_vector)
        sensor_vec = "".join(str(int(v)) for v in predicate_vector)

        for mode, transitions in self.automaton_dict.items():
            if False: 
                print("mode:", mode)
                print("transitions:", transitions)
            for pattern, target in transitions.items():
                if False: 
                    print("  pattern:", pattern, " target:", target)    
                match_ = self._match_pattern(sensor_vec, pattern)
                if False: 
                    print("sensor: ", sensor_vec)
                    print("pattern: ", pattern)
                    print(match_)
                if self._match_pattern(sensor_vec, pattern):
                    return target  # this is the mode we would transition to

        return None  # no match found
    
    def set_predicates(self, predicate_vector, update_mode=True):
        """
        Directly set the predicate vector.
        Optionally update the current mode based on the new predicates.
        Returns True if a mode was found and set, False otherwise.
        """
        print("[Backdoor] Setting predicates to:", predicate_vector)
        self.predicates = np.array(predicate_vector, dtype=int)
        if update_mode:
            mode = self.get_mode_from_predicates(self.predicates)
            if mode is not None:
                self.curr_mode = mode
                self.prev_preds = self.predicates.copy()
                print(f"{COLOR_BLUE}Backdoor: set mode to {mode} from predicate vector {predicate_vector}{COLOR_RESET}")
                return True
            else:
                print(f"{COLOR_RED}Backdoor: no valid mode for predicate vector {predicate_vector}{COLOR_RESET}")
                return False
        return True

    def get_current_predicates(self):
        """Return the current predicate vector."""
        return self.predicates.copy() if self.predicates is not None else None
    
    def get_current_mode(self):
        """Return the current mode of the automaton."""
        return self.curr_mode
    
if __name__ == "__main__":
    monitor = LTLMonitor()
    # Example usage
    print("monitor.automaton_dict: ", monitor.automaton_dict)
    for mode, transitions in monitor.automaton_dict.items():
        print("mode:", mode)
        print("transitions:", transitions)
        for pattern, target in transitions.items():
            print("  pattern:", pattern, " target:", target)    
        
    print("done")
   