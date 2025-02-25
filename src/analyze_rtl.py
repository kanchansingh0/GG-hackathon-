import pandas as pd
import joblib

def analyze_rtl_complexity():
    # Circuit characteristics
    circuit_metrics = {
        'setup_slack': 7.5,        # Tight setup timing due to complex multiplication
        'hold_slack': 0.2,         # Potential hold violation in output logic
        'timing_violation': 1,     # Has timing violations
        'fanin_count': 6,         # Multiple inputs affecting output
        'fanout_count': 4,        # Output feeds multiple destinations
        'path_length': 8          # Long path through multiplication logic
    }
    
    # Create DataFrame
    input_data = pd.DataFrame([circuit_metrics])
    
    # Calculate derived features
    input_data['slack_ratio'] = input_data['setup_slack'] / (input_data['hold_slack'].abs() + 1e-6)
    input_data['complexity_score'] = input_data['fanin_count'] * input_data['fanout_count']
    input_data['path_density'] = input_data['path_length'] / (input_data['fanout_count'] + 1)
    input_data['timing_score'] = input_data['setup_slack'] * (1 - input_data['timing_violation'])
    
    try:
        # Load model and make prediction
        model = joblib.load('models/improved_rf_model.joblib')
        complexity = model.predict(input_data)[0]
        
        print("\nRTL Circuit Analysis:")
        print("=====================")
        print("\nCircuit Issues:")
        print("1. Complex multiplication logic with long paths")
        print("2. Multiple shifting operations with variable shift amounts")
        print("3. Potential hold violations in output logic")
        print("4. Setup timing constraints in multiplication path")
        
        print("\nCircuit Metrics:")
        for metric, value in circuit_metrics.items():
            print(f"{metric}: {value}")
        
        print(f"\nPredicted Circuit Complexity: {complexity:.4f}")
        
        print("\nRecommendations:")
        if complexity > 7.0:
            print("- Consider pipelining the multiplication logic")
            print("- Add registers to break long combinational paths")
            print("- Review clock constraints and timing requirements")
        elif complexity > 5.0:
            print("- Optimize multiplication logic")
            print("- Review shift operations implementation")
            print("- Consider adding pipeline registers")
        else:
            print("- Circuit complexity is manageable")
            print("- Monitor timing margins during implementation")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    analyze_rtl_complexity() 