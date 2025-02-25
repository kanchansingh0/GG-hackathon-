import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(confusion_matrix: Dict, save_path: str = None):
    """Plot confusion matrix"""
    cm_values = np.array([
        [confusion_matrix['true_negative'], confusion_matrix['false_positive']],
        [confusion_matrix['false_negative'], confusion_matrix['true_positive']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_values, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Violation', 'Violation'],
                yticklabels=['No Violation', 'Violation'])
    plt.title('Timing Violation Detection\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(feature_names: List[str], importance: np.ndarray, save_path: str = None):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Feature Importance in Timing Prediction')
    plt.xlabel('Importance')
    
    if save_path:
        plt.savefig(save_path)
    plt.show() 