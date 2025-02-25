import re
from pathlib import Path
from typing import Dict, List

class RTLExtractor:
    def __init__(self, rtl_file_path: str):
        self.rtl_file_path = Path(rtl_file_path)
        self.module_content = ""
        self.signals = {}
        
    def read_rtl_file(self) -> str:
        """Read RTL file content"""
        with open(self.rtl_file_path, 'r') as f:
            self.module_content = f.read()
        return self.module_content
    
    def extract_signals(self) -> Dict[str, Dict]:
        """Extract all signals and their properties from RTL"""
        # Extract signal declarations
        reg_pattern = r'reg\s+(\[[\d:]+\])?\s*(\w+)'
        wire_pattern = r'wire\s+(\[[\d:]+\])?\s*(\w+)'
        
        # Find all reg and wire declarations
        reg_matches = re.finditer(reg_pattern, self.module_content)
        wire_matches = re.finditer(wire_pattern, self.module_content)
        
        for match in reg_matches:
            width, name = match.groups()
            self.signals[name] = {
                'type': 'reg',
                'width': width if width else '[0:0]',
                'fanin': [],
                'logic_depth': 0
            }
            
        for match in wire_matches:
            width, name = match.groups()
            self.signals[name] = {
                'type': 'wire',
                'width': width if width else '[0:0]',
                'fanin': [],
                'logic_depth': 0
            }
            
        return self.signals
    
    def analyze_logic_depth(self, signal_name: str) -> int:
        """Analyze combinational logic depth for a signal"""
        if signal_name not in self.signals:
            return 0
            
        # Find all assignments to this signal
        assign_pattern = f"assign\s+{signal_name}\s*=\s*(.+?);"
        always_pattern = f"always\s*@.*?\s*{signal_name}\s*<=\s*(.+?);"
        
        assign_matches = re.finditer(assign_pattern, self.module_content)
        always_matches = re.finditer(always_pattern, self.module_content)
        
        max_depth = 0
        for match in list(assign_matches) + list(always_matches):
            expression = match.group(1)
            depth = self._calculate_expression_depth(expression)
            max_depth = max(max_depth, depth)
            
        self.signals[signal_name]['logic_depth'] = max_depth
        return max_depth
    
    def _calculate_expression_depth(self, expression: str) -> int:
        """Calculate logic depth of an expression"""
        # Count basic operations
        ops = re.findall(r'[&|^+*/-]', expression)
        # Count nested parentheses depth
        max_paren_depth = 0
        current_depth = 0
        for char in expression:
            if char == '(':
                current_depth += 1
                max_paren_depth = max(max_paren_depth, current_depth)
            elif char == ')':
                current_depth -= 1
                
        return len(ops) + max_paren_depth
