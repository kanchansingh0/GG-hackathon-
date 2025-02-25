import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

class SyntheticRTLGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_module(self, module_name: str, complexity: int) -> Tuple[str, Dict]:
        """Generate synthetic RTL module with known logic depth"""
        signals = []
        connections = []
        actual_depths = {}
        
        # Generate input signals
        input_signals = [f"in_{i}" for i in range(random.randint(3, 8))]
        signals.extend(input_signals)
        
        # Generate intermediate signals with known depth
        for i in range(complexity):
            sig_name = f"wire_{i}"
            signals.append(sig_name)
            
            # Create random logic expression
            inputs = random.sample(signals[:-1], random.randint(2, 4))
            expr = self._generate_logic_expr(inputs)
            connections.append(f"assign {sig_name} = {expr};")
            
            # Calculate actual depth
            depth = max(actual_depths.get(s, 0) for s in inputs) + 1
            actual_depths[sig_name] = depth
        
        # Generate output signal
        output_signal = "out"
        signals.append(output_signal)
        inputs = random.sample(signals[:-1], random.randint(2, 4))
        expr = self._generate_logic_expr(inputs)
        connections.append(f"assign {output_signal} = {expr};")
        actual_depths[output_signal] = max(actual_depths.get(s, 0) for s in inputs) + 1
        
        # Create RTL content
        rtl_content = self._create_rtl_module(module_name, input_signals, [output_signal], signals, connections)
        
        return rtl_content, actual_depths
    
    def _generate_logic_expr(self, inputs: List[str]) -> str:
        """Generate random logic expression"""
        ops = ['&', '|', '^']
        expr = inputs[0]
        for inp in inputs[1:]:
            op = random.choice(ops)
            expr = f"({expr} {op} {inp})"
        return expr
    
    def _create_rtl_module(self, name: str, inputs: List[str], outputs: List[str], 
                          signals: List[str], connections: List[str]) -> str:
        """Create complete RTL module"""
        module = [f"module {name}("]
        
        # Ports
        ports = [f"input {s}" for s in inputs] + [f"output {s}" for s in outputs]
        module.append("    " + ",\n    ".join(ports))
        module.append(");")
        
        # Signal declarations
        for sig in signals:
            if sig not in inputs and sig not in outputs:
                module.append(f"wire {sig};")
        
        # Logic
        module.extend(connections)
        module.append("endmodule")
        
        return "\n".join(module)
    
    def generate_dataset(self, num_modules: int, complexity_range: Tuple[int, int]) -> Dict:
        """Generate multiple RTL modules with varying complexity"""
        dataset = {}
        
        for i in range(num_modules):
            complexity = random.randint(*complexity_range)
            module_name = f"test_module_{i}"
            
            # Generate RTL and actual depths
            rtl_content, actual_depths = self.generate_module(module_name, complexity)
            
            # Save RTL file
            rtl_path = self.output_dir / f"{module_name}.v"
            with open(rtl_path, 'w') as f:
                f.write(rtl_content)
            
            dataset[module_name] = {
                'rtl_path': str(rtl_path),
                'actual_depths': actual_depths
            }
        
        return dataset

class SignalGenerator:
    def __init__(self):
        self.supported_ops = ['and', 'or', 'xor', 'add', 'mult']
        self.gate_delays = {
            'and': 0.1,
            'or': 0.1,
            'xor': 0.15,
            'add': 0.3,
            'mult': 0.8
        }
    
    def generate_signal_path(self, depth: int) -> Dict:
        """Generate a synthetic signal path with known timing characteristics"""
        path = {
            'operations': [],
            'gate_count': 0,
            'critical_path_delay': 0.0,
            'fanin_signals': []
        }
        
        # Generate random operations for the path
        for _ in range(depth):
            op = random.choice(self.supported_ops)
            path['operations'].append(op)
            path['gate_count'] += self._get_gate_count(op)
            path['critical_path_delay'] += self.gate_delays[op]
            
            # Add random fanin signals
            num_fanin = random.randint(2, 4)
            fanin_signals = [f"sig_{random.randint(0, 100)}" for _ in range(num_fanin)]
            path['fanin_signals'].extend(fanin_signals)
        
        return path
    
    def generate_timing_data(self, path: Dict, clock_period: float) -> Dict:
        """Generate timing analysis data for a path"""
        delay = path['critical_path_delay']
        setup_time = 0.1  # Standard setup time
        
        timing_data = {
            'path_delay': delay,
            'setup_slack': clock_period - delay - setup_time,
            'has_violation': delay + setup_time > clock_period,
            'gate_count': path['gate_count'],
            'operation_types': list(set(path['operations']))
        }
        
        return timing_data
    
    def _get_gate_count(self, operation: str) -> int:
        """Get gate count for an operation"""
        gate_counts = {
            'and': 1,
            'or': 1,
            'xor': 2,
            'add': 5,
            'mult': 20
        }
        return gate_counts.get(operation, 1)
