import yaml
import re
import numpy as np

class LogicEnforcer:
    def __init__(self, predicate_to_idx, yaml_path):
        self.predicate_to_idx = predicate_to_idx
        # Invert map for lookups
        self.idx_to_predicate = {v: k for k, v in predicate_to_idx.items()}
        self.PRED_DIM = len(predicate_to_idx)
        
        # Pre-compile the regex for parsing "name(arg1, arg2)"
        self.pred_regex = re.compile(r"(\w+)\((.*)\)")
        
        # Structure the predicates for fast lookup: {'on': {('a', 'b'): 12}}
        self.structured_preds = self._structure_predicates()
        
        with open(yaml_path, 'r') as f:
            self.rules = yaml.safe_load(f)['rules']
        # print(f"self.rules {self.rules}")

    def _structure_predicates(self):
        structured = {}
        for pred_str, idx in self.predicate_to_idx.items():
            match = self.pred_regex.match(pred_str)
            if match:
                name = match.group(1)
                args = tuple(x.strip() for x in match.group(2).split(','))
                
                if name not in structured:
                    structured[name] = {}
                structured[name][args] = idx
        return structured

    def _get_idx(self, pred_name, args):
        """Helper to safely get index if it exists in the domain."""
        if pred_name in self.structured_preds and args in self.structured_preds[pred_name]:
            return self.structured_preds[pred_name][args]
        return None
    
    def _parse_atom(self, s):
        # Parses "name(arg1, arg2)" into "name", ["arg1", "arg2"]
        m = re.match(r"(\w+)\((.*)\)", s)
        if m:
            args = [x.strip() for x in m.group(2).split(',')]
            return m.group(1), args
        return None, []
    
    def apply_rules(self, pred_vector):
        """
        Applies rules iteratively until the vector stops changing.
        """
        current_vec = pred_vector.copy()
        
        # Debug: Show active predicates at start
        active_indices = np.where(current_vec == 1)[0]
        active_names = [self.idx_to_predicate[i] for i in active_indices]
        if active_names:
            print(f"\n--- START LOGIC ENFORCEMENT ---")
            print(f"Active Inputs: {active_names}")
        
        max_iterations = 10 
        
        for i in range(max_iterations):
            new_vec = current_vec.copy()
            
            for rule_idx, rule in enumerate(self.rules):
                if rule['type'] == 'implication':
                    self._apply_implication(rule, new_vec, rule_idx)
                elif rule['type'] == 'conditional_implication':
                    self._apply_conditional(rule, new_vec, rule_idx)
                elif rule['type'] == 'exclusion':
                    self._apply_exclusion(rule, new_vec, rule_idx)

            if np.array_equal(new_vec, current_vec):
                break 
            
            current_vec = new_vec
            print(f"*** iteration {i}/{max_iterations}")
        return current_vec

    # --- Rule Logic Helpers ---

    def _apply_implication(self, rule, vec, rule_idx):
        # Trigger: grasped(a) -> t_name='grasped', t_vars=['a']
        t_name, t_vars = self._parse_atom(rule['if'])
        e_name, e_vars = self._parse_atom(rule['then'])
        
        if t_name in self.structured_preds:
            for t_args, t_idx in self.structured_preds[t_name].items():
                if vec[t_idx] == 1:
                    # FIX: t_vars is a list ['a'], so we must use t_vars[0] as key
                    mapping = {t_vars[0]: t_args[0]} 
                    
                    resolved_args = tuple(mapping.get(v, v) for v in e_vars)
                    e_idx = self._get_idx(e_name, resolved_args)
                    
                    if e_idx is not None:
                        # 1. Apply Rule
                        if vec[e_idx] == 0:
                            print(f"   [RULE {rule_idx}] {self.idx_to_predicate[t_idx]} is True")
                            print(f"      -> Setting {self.idx_to_predicate[e_idx]} = 1")
                            vec[e_idx] = 1

                        # 2. CONFLICT FIX
                        if e_name == 'above':
                            on_idx = self._get_idx('on', resolved_args)
                            if on_idx is not None and vec[on_idx] == 1:
                                print(f"      -> [CONFLICT] Clearing {self.idx_to_predicate[on_idx]}")
                                vec[on_idx] = 0

    def _apply_conditional(self, rule, vec, rule_idx):
        t_name, t_vars = self._parse_atom(rule['trigger'])
        c_name, c_vars = self._parse_atom(rule['condition'])
        e_name, e_vars = self._parse_atom(rule['effect'])

        if t_name in self.structured_preds:
            for t_args, t_idx in self.structured_preds[t_name].items():
                if vec[t_idx] == 1:
                    mapping = dict(zip(t_vars, t_args))
                    
                    if c_name in self.structured_preds:
                        for c_args, c_idx in self.structured_preds[c_name].items():
                            if vec[c_idx] == 1:
                                # Relies on 'a' being the shared variable name in YAML
                                if 'a' in c_vars:
                                    a_idx_in_cond = c_vars.index('a') 
                                    if c_args[a_idx_in_cond] == mapping['a']:
                                        c_var_name = c_vars[1 - a_idx_in_cond] 
                                        mapping[c_var_name] = c_args[1 - a_idx_in_cond]
                                        
                                        resolved_args = tuple(mapping.get(v, v) for v in e_vars)
                                        e_idx = self._get_idx(e_name, resolved_args)
                                        
                                        if e_idx is not None:
                                            # 1. Apply Rule
                                            if vec[e_idx] == 0:
                                                print(f"   [CHAIN {rule_idx}] {self.idx_to_predicate[t_idx]} AND {self.idx_to_predicate[c_idx]}")
                                                print(f"      -> Setting {self.idx_to_predicate[e_idx]} = 1")
                                                vec[e_idx] = 1

                                            # 2. CONFLICT FIX
                                            if e_name == 'above':
                                                on_idx = self._get_idx('on', resolved_args)
                                                if on_idx is not None and vec[on_idx] == 1:
                                                    print(f"      -> [CONFLICT FIX] Clearing {self.idx_to_predicate[on_idx]}")
                                                    vec[on_idx] = 0

    def _apply_exclusion(self, rule, vec, rule_idx):
        t_name, t_vars = self._parse_atom(rule['if'])
        e_name, e_vars = self._parse_atom(rule['then_not'])
        
        if t_name in self.structured_preds:
            for t_args, t_idx in self.structured_preds[t_name].items():
                if vec[t_idx] == 1:
                    mapping = dict(zip(t_vars, t_args))
                    resolved_args = tuple(mapping.get(v, v) for v in e_vars)
                    e_idx = self._get_idx(e_name, resolved_args)
                    
                    if e_idx is not None:
                        if vec[e_idx] == 1:
                            print(f"   [EXCLUSION {rule_idx}] {self.idx_to_predicate[t_idx]} is True")
                            print(f"      -> Forcing {self.idx_to_predicate[e_idx]} to 0")
                            vec[e_idx] = 0