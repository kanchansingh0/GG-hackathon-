from pathlib import Path
import json
import random
from typing import Dict, List

class SyntheticDataManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.rtl_dir = self.base_dir / "raw" / "synthetic" / "rtl_modules"
        self.timing_dir = self.base_dir / "raw" / "synthetic" / "timing_reports"
        
        # Create directories if they don't exist
        self.rtl_dir.mkdir(parents=True, exist_ok=True)
        self.timing_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_rtl(self, module_name: str, complexity: int) -> str:
        """Generate synthetic RTL with known timing characteristics"""
        rtl_content = [
            f"module {module_name} (",
            "    input clk,",
            "    input rst_n,"
        ]
        
        # Generate inputs
        num_inputs = random.randint(2, 5)
        inputs = [f"    input [{random.randint(0, 32)}:0] in_{i}" for i in range(num_inputs)]
        rtl_content.extend(inputs)
        
        # Generate outputs
        num_outputs = random.randint(1, 3)
        outputs = [f"    output reg [{random.randint(0, 32)}:0] out_{i}" for i in range(num_outputs)]
        rtl_content.extend(outputs)
        
        rtl_content.append(");")
        
        # Generate internal signals
        for i in range(complexity):
            rtl_content.append(f"    wire [{random.randint(0, 32)}:0] wire_{i};")
        
        # Generate combinational logic
        ops = ['&', '|', '^', '+', '*']
        for i in range(complexity):
            inputs = [f"in_{random.randint(0, num_inputs-1)}" for _ in range(2)]
            op = random.choice(ops)
            rtl_content.append(f"    assign wire_{i} = {inputs[0]} {op} {inputs[1]};")
        
        # Generate sequential logic
        rtl_content.extend([
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) begin"
        ])
        
        for i in range(num_outputs):
            rtl_content.append(f"            out_{i} <= 0;")
        
        rtl_content.append("        end else begin")
        
        for i in range(num_outputs):
            wire_idx = random.randint(0, complexity-1)
            rtl_content.append(f"            out_{i} <= wire_{wire_idx};")
        
        rtl_content.extend([
            "        end",
            "    end",
            "endmodule"
        ])
        
        return "\n".join(rtl_content)
    
    def save_rtl_file(self, module_name: str, rtl_content: str) -> Path:
        """Save RTL content to file"""
        file_path = self.rtl_dir / f"{module_name}.v"
        with open(file_path, 'w') as f:
            f.write(rtl_content)
        return file_path
    
    def save_timing_report(self, module_name: str, timing_data: Dict) -> Path:
        """Save timing report"""
        file_path = self.timing_dir / f"{module_name}_timing.json"
        with open(file_path, 'w') as f:
            json.dump(timing_data, f, indent=2)
        return file_path
    
    def generate_dataset(self, num_modules: int, complexity_range: tuple) -> Dict:
        """Generate complete synthetic dataset"""
        dataset = {}
        
        for i in range(num_modules):
            module_name = f"test_module_{i}"
            complexity = random.randint(*complexity_range)
            
            # Generate and save RTL
            rtl_content = self.generate_synthetic_rtl(module_name, complexity)
            rtl_path = self.save_rtl_file(module_name, rtl_content)
            
            # Generate timing data
            timing_data = {
                'module': module_name,
                'complexity': complexity,
                'critical_path_depth': complexity + random.randint(1, 3),
                'timing_violations': random.random() > 0.7
            }
            timing_path = self.save_timing_report(module_name, timing_data)
            
            dataset[module_name] = {
                'rtl_path': str(rtl_path),
                'timing_path': str(timing_path),
                'complexity': complexity
            }
        
        return dataset

if __name__ == "__main__":
    # Create an instance of SyntheticDataManager
    data_manager = SyntheticDataManager(base_dir="data")
    
    # Generate a small dataset (5 modules with complexity between 3 and 7)
    dataset = data_manager.generate_dataset(
        num_modules=5,
        complexity_range=(3, 7)
    )
    
    # Print the generated dataset information
    print("\nGenerated Dataset:")
    for module_name, info in dataset.items():
        print(f"\nModule: {module_name}")
        print(f"RTL Path: {info['rtl_path']}")
        print(f"Timing Path: {info['timing_path']}")
        print(f"Complexity: {info['complexity']}") 