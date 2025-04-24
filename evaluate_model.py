from ultralytics import YOLO
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_model(model_path, data_yaml=None, save_results=True):
    """
    Evaluate the YOLOv8 model and display accuracy metrics with an option to save results.
    
    Parameters:
        model_path (str): Path to the trained model file (.pt)
        data_yaml (str): Path to the data configuration file (optional)
        save_results (bool): Save evaluation results to a file and chart
    
    Returns:
        metrics: Object containing evaluation metrics, or None in case of an error
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at path: {model_path}")
        return
    
    try:
        # Load the model
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        
        # Evaluate the model
        print("Evaluating the model...")
        if data_yaml:
            # Use the data configuration file if specified
            metrics = model.val(data=data_yaml)
        else:
            # Use the default dataset associated with the model
            metrics = model.val()
        
        # Display evaluation results
        print("\n📈 Evaluation Results:")
        print(f"Precision: {metrics.box.p:.4f}")
        print(f"Recall: {metrics.box.r:.4f}")
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        
        # Calculate overall accuracy (Accuracy)
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        # In object detection, it can be approximated using F1-score
        f1_score = 2 * (metrics.box.p * metrics.box.r) / (metrics.box.p + metrics.box.r + 1e-16)
        print(f"F1 Score (Approximation of overall accuracy): {f1_score:.4f}")
        
        # Save results if requested
        if save_results:
            save_evaluation_results(metrics, model_path)
        
        return metrics
        
    except Exception as e:
        print(f"An error occurred while evaluating the model: {str(e)}")
        print("\nSuggestions for fixing the problem:")
        print("1. Ensure the model file (.pt) is not corrupted")
        print("2. Make sure you have the correct version of the ultralytics library")
        print("3. Verify that the evaluation dataset is in the correct path")
        
        # Check the model file size to ensure it is not corrupted
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # in megabytes
            print(f"\nModel file size: {file_size:.2f} MB")
            if file_size < 5:  # Model files are usually larger than 5 MB
                print("Warning: Model file is very small, it may be corrupted or incomplete")
        
        return None

def save_evaluation_results(metrics, model_path):
    """
    Save evaluation results to a text file and chart.
    
    Parameters:
        metrics: Object containing evaluation metrics
        model_path: Path of the model file used for evaluation
    """
    # Create a directory for results if it doesn't exist
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Extract model name from the path
    model_name = Path(model_path).stem
    
    # Save results to a text file
    results_file = results_dir / f"{model_name}_evaluation.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Results for Model: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Precision: {metrics.box.p:.4f}\n")
        f.write(f"Recall: {metrics.box.r:.4f}\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        
        # Calculate F1 score
        f1_score = 2 * (metrics.box.p * metrics.box.r) / (metrics.box.p + metrics.box.r + 1e-16)
        f.write(f"F1 Score (Approximation of overall accuracy): {f1_score:.4f}\n")
    
    print(f"Saved evaluation results in: {results_file}")
    
    # Create a chart for the results
    create_evaluation_chart(metrics, model_name, results_dir)

def create_evaluation_chart(metrics, model_name, results_dir):
    """
    Create a chart for evaluation results.
    
    Parameters:
        metrics: Object containing evaluation metrics
        model_name: Name of the model
        results_dir: Directory to save results
    """
    # Extract values
    precision = metrics.box.p
    recall = metrics.box.r
    map50 = metrics.box.map50
    map = metrics.box.map
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)
    
    # Create the chart
    plt.figure(figsize=(10, 6))
    metrics_names = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'F1-Score']
    metrics_values = [precision, recall, map50, map, f1_score]
    
    # Draw the bar chart
    bars = plt.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'])
    
    # Add values above the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Format the chart
    plt.ylim(0, 1.1)
    plt.title(f'Evaluation Metrics for Model {model_name}', fontsize=16)
    plt.ylabel('Value', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the chart
    chart_file = results_dir / f"{model_name}_evaluation_chart.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved chart in: {chart_file}")

def analyze_player_performance(model, video_path, output_path=None, confidence=0.25):
    """
    Analyze player performance in a video using YOLOv8 model.
    
    Parameters:
        model: Trained YOLO model
        video_path: Path to the video file
        output_path: Path to save the processed video (optional)
        confidence: Confidence threshold for detection
    
    Returns:
        Analysis results
    """
    print(f"Analyzing player performance in video: {video_path}")
    
    # Perform tracking on the video
    results = model.track(
        source=video_path,
        conf=confidence,
        save=True if output_path else False,
        project=Path(output_path).parent if output_path else None,
        name=Path(output_path).stem if output_path else None,
        tracker="bytetrack"  # Use ByteTrack algorithm for tracking
    )
    
    print("Video analysis completed")
    if output_path:
        print(f"Saved processed video in: {output_path}")
    
    return results

if __name__ == "__main__":
    # User can specify the model path and data configuration file as command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        data_yaml = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Use default values if no arguments are specified
        model_path = r"C:\Users\SHAHAD\OneDrive\Desktop\ScoutX\sports-ai-env\yolov8x.pt"  # Change this to the correct path
        data_yaml = "yolo_data.yaml"  # Change this to the correct path
    
    # Evaluate the model
    metrics = evaluate_model(model_path, data_yaml, save_results=True)
    
    # If evaluation is successful and a video path is specified, analyze player performance
    if metrics and len(sys.argv) > 3:
        video_path = sys.argv[3]
        output_path = sys.argv[4] if len(sys.argv) > 4 else None
        
        # Load the model again for analysis
        model = YOLO(model_path)
        analyze_player_performance(model, video_path, output_path)