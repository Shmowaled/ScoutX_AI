import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure paths
OUTPUT_DIR = "player_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_sample_images():
    """
    Download sample football images for analysis.
    """
    print("Downloading sample football images...")
    
    # URLs for football images from specified datasets
    image_urls = [
        "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg",
        "https://github.com/ultralytics/assets/raw/main/im/bus.jpg",
        "https://github.com/ultralytics/assets/raw/main/im/soccer.jpg"
    ]
    
    images = []
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img_path = os.path.join(OUTPUT_DIR, f"sample_image_{i+1}.jpg")
            img.save(img_path)
            images.append(img_path)
            print(f"Downloaded image {i+1} and saved to {img_path}")
        except Exception as e:
            print(f"Failed to download image from {url}: {str(e)}")
    
    return images

def simulate_player_detection(image_path):
    """
    Simulate player detection in an image.
    
    Parameters:
        image_path: Path to the image.
        
    Returns:
        List of detected players' information (locations and IDs).
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image from {image_path}")
        return []
    
    height, width = img.shape[:2]
    
    # Simulate player detection (in a real application, a YOLO model would be used here)
    num_players = np.random.randint(5, 15)  # Random number of players
    
    players = []
    for i in range(num_players):
        # Create a random bounding box
        x1 = np.random.randint(0, width - 100)
        y1 = np.random.randint(0, height - 200)
        w = np.random.randint(50, 100)
        h = np.random.randint(100, 200)
        x2 = x1 + w
        y2 = y1 + h
        
        # Assign a random team ID (0 or 1)
        team_id = np.random.randint(0, 2)
        
        # Create player information
        player = {
            "id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "team_id": team_id,
            "confidence": np.random.uniform(0.7, 0.99)
        }
        
        players.append(player)
    
    return players

def visualize_player_detection(image_path, players, output_path):
    """
    Create an image showing player locations.
    
    Parameters:
        image_path: Path to the image.
        players: Players' information.
        output_path: Path to save the image.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image from {image_path}")
        return
    
    # Copy of the image for drawing
    img_draw = img.copy()
    
    # Team colors
    team_colors = [(0, 0, 255), (255, 0, 0)]  # Red and Blue
    
    # Draw rectangles around players
    for player in players:
        bbox = player["bbox"]
        team_id = player["team_id"]
        player_id = player["id"]
        confidence = player["confidence"]
        
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), team_colors[team_id], 2)
        
        # Add player ID
        label = f"Player {player_id} ({confidence:.2f})"
        cv2.putText(img_draw, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_colors[team_id], 2)
    
    # Save the image
    cv2.imwrite(output_path, img_draw)
    
    print(f"Created player detection image at: {output_path}")
    print(f"Number of detected players: {len(players)}")

def track_player_movements(image_paths):
    """
    Simulate tracking player movements across a series of images.
    
    Parameters:
        image_paths: List of image paths.
        
    Returns:
        Player tracking data.
    """
    print("Simulating player movement tracking...")
    
    all_tracks = {}
    
    for frame_idx, image_path in enumerate(image_paths):
        # Detect players in the current frame
        players = simulate_player_detection(image_path)
        
        # Update player tracks
        for player in players:
            player_id = player["id"]
            
            if player_id not in all_tracks:
                all_tracks[player_id] = {
                    "frames": [],
                    "positions": [],
                    "team_id": player["team_id"]
                }
            
            # Extract player center position
            bbox = player["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Add frame and position information
            all_tracks[player_id]["frames"].append(frame_idx)
            all_tracks[player_id]["positions"].append((center_x, center_y))
    
    return all_tracks

def calculate_player_statistics(tracks):
    """
    Calculate player performance statistics from tracking data.
    
    Parameters:
        tracks: Player tracking data.
        
    Returns:
        Player statistics.
    """
    print("Calculating player performance statistics...")
    
    player_stats = {}
    
    for player_id, track_data in tracks.items():
        positions = track_data["positions"]
        team_id = track_data["team_id"]
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]
            x2, y2 = positions[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        # Calculate average speed (distance per frame)
        avg_speed = total_distance / len(positions) if len(positions) > 0 else 0
        
        # Calculate average position (mean location)
        avg_x = np.mean([p[0] for p in positions]) if positions else 0
        avg_y = np.mean([p[1] for p in positions]) if positions else 0
        
        # Calculate coverage area (standard deviation of positions)
        std_x = np.std([p[0] for p in positions]) if len(positions) > 1 else 0
        std_y = np.std([p[1] for p in positions]) if len(positions) > 1 else 0
        coverage_area = std_x * std_y
        
        # Store statistics
        player_stats[player_id] = {
            "team_id": team_id,
            "total_distance": total_distance,
            "avg_speed": avg_speed,
            "avg_position": (avg_x, avg_y),
            "coverage_area": coverage_area
        }
    
    return player_stats

def create_heatmap(tracks, image_path, output_path):
    """
    Create a heatmap of player locations.
    
    Parameters:
        tracks: Player tracking data.
        image_path: Path to the original image.
        output_path: Path to save the heatmap.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image from {image_path}")
        return
    
    height, width = img.shape[:2]
    
    # Create a matrix to store density data
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Add points for each player location
    for player_id, track_data in tracks.items():
        positions = track_data["positions"]
        
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            
            # Ensure coordinates are within image bounds
            if 0 <= x < width and 0 <= y < height:
                # Add a heat point with Gaussian effect
                sigma = 30  # Spread of the effect
                for i in range(max(0, y-sigma*3), min(height, y+sigma*3)):
                    for j in range(max(0, x-sigma*3), min(width, x+sigma*3)):
                        dist = np.sqrt((i-y)**2 + (j-x)**2)
                        influence = np.exp(-(dist**2) / (2*sigma**2))
                        heatmap[i, j] += influence
    
    # Normalize the heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    
    # Apply color map
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    alpha = 0.6
    overlay = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    
    # Save the image
    cv2.imwrite(output_path, overlay)
    
    print(f"Created heatmap at: {output_path}")

def create_player_performance_chart(player_stats, output_path):
    """
    Create a chart for player performance.
    
    Parameters:
        player_stats: Player statistics.
        output_path: Path to save the chart.
    """
    # Convert data to DataFrame
    data = []
    for player_id, stats in player_stats.items():
        data.append({
            "Player ID": f"Player {player_id}",
            "Team": f"Team {stats['team_id'] + 1}",
            "Distance": stats["total_distance"],
            "Speed": stats["avg_speed"],
            "Coverage": stats["coverage_area"]
        })
    
    df = pd.DataFrame(data)
    
    # Create multi-chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Distance bar plot
    sns.barplot(x="Player ID", y="Distance", hue="Team", data=df, ax=axes[0])
    axes[0].set_title("Distance Covered by Each Player")
    axes[0].set_ylabel("Distance")
    axes[0].set_xlabel("Player ID")
    
    # Speed bar plot
    sns.barplot(x="Player ID", y="Speed", hue="Team", data=df, ax=axes[1])
    axes[1].set_title("Average Speed of Each Player")
    axes[1].set_ylabel("Speed")
    axes[1].set_xlabel("Player ID")
    
    # Coverage area bar plot
    sns.barplot(x="Player ID", y="Coverage", hue="Team", data=df, ax=axes[2])
    axes[2].set_title("Coverage Area of Each Player")
    axes[2].set_ylabel("Coverage Area")
    axes[2].set_xlabel("Player ID")
    
    # Format the chart
    plt.tight_layout()
    
    # Save chart
    plt.savefig(output_path)
    
    print(f"Created player performance chart at: {output_path}")

def create_team_comparison_chart(player_stats, output_path):
    """
    Create a chart comparing the performance of two teams.
    
    Parameters:
        player_stats: Player statistics.
        output_path: Path to save the chart.
    """
    # Aggregate team statistics
    team_stats = {0: {"distance": 0, "speed": 0, "coverage": 0, "count": 0},
                   1: {"distance": 0, "speed": 0, "coverage": 0, "count": 0}}
    
    for player_id, stats in player_stats.items():
        team_id = stats["team_id"]
        team_stats[team_id]["distance"] += stats["total_distance"]
        team_stats[team_id]["speed"] += stats["avg_speed"]
        team_stats[team_id]["coverage"] += stats["coverage_area"]
        team_stats[team_id]["count"] += 1
    
    # Calculate averages
    for team_id in team_stats:
        if team_stats[team_id]["count"] > 0:
            team_stats[team_id]["avg_distance"] = team_stats[team_id]["distance"] / team_stats[team_id]["count"]
            team_stats[team_id]["avg_speed"] = team_stats[team_id]["speed"] / team_stats[team_id]["count"]
            team_stats[team_id]["avg_coverage"] = team_stats[team_id]["coverage"] / team_stats[team_id]["count"]
    
    # Create DataFrame for the chart
    data = []
    for team_id, stats in team_stats.items():
        if stats["count"] > 0:
            data.append({
                "Team": f"Team {team_id + 1}",
                "Metric": "Distance",
                "Value": stats["avg_distance"]
            })
            data.append({
                "Team": f"Team {team_id + 1}",
                "Metric": "Speed",
                "Value": stats["avg_speed"]
            })
            data.append({
                "Team": f"Team {team_id + 1}",
                "Metric": "Coverage",
                "Value": stats["avg_coverage"]
            })
    
    df = pd.DataFrame(data)
    
    # Create a comparison chart
    plt.figure(figsize=(12, 8))
    
    # Comparison bar plot
    sns.barplot(x="Metric", y="Value", hue="Team", data=df)
    
    # Format the chart
    plt.title("Comparison of Team Performance")
    plt.ylabel("Value")
    plt.xlabel("Metric")
    
    # Save chart
    plt.savefig(output_path)
    
    print(f"Created team comparison chart at: {output_path}")

def create_interactive_dashboard(player_stats, tracks, output_path):
    """
    Create an interactive dashboard for player performance.
    
    Parameters:
        player_stats: Player statistics.
        tracks: Player tracking data.
        output_path: Path to save the dashboard.
    """
    # Convert data to DataFrame
    data = []
    for player_id, stats in player_stats.items():
        data.append({
            "Player ID": f"Player {player_id}",
            "Team": f"Team {stats['team_id'] + 1}",
            "Distance": stats["total_distance"],
            "Speed": stats["avg_speed"],
            "Coverage": stats["coverage_area"],
            "X Position": stats["avg_position"][0],
            "Y Position": stats["avg_position"][1]
        })
    
    df = pd.DataFrame(data)
    
    # Create interactive dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Distance Covered", "Average Speed", "Coverage Area", "Player Locations"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Add distance bar plot
    fig.add_trace(
        go.Bar(x=df["Player ID"], y=df["Distance"], name="Distance", 
               marker_color=df["Team"].map({"Team 1": "red", "Team 2": "blue"})),
        row=1, col=1
    )
    
    # Add speed bar plot
    fig.add_trace(
        go.Bar(x=df["Player ID"], y=df["Speed"], name="Speed",
               marker_color=df["Team"].map({"Team 1": "red", "Team 2": "blue"})),
        row=1, col=2
    )
    
    # Add coverage area bar plot
    fig.add_trace(
        go.Bar(x=df["Player ID"], y=df["Coverage"], name="Coverage",
               marker_color=df["Team"].map({"Team 1": "red", "Team 2": "blue"})),
        row=2, col=1
    )
    
    # Add player locations scatter plot
    for team in ["Team 1", "Team 2"]:
        team_df = df[df["Team"] == team]
        fig.add_trace(
            go.Scatter(x=team_df["X Position"], y=team_df["Y Position"], 
                       mode="markers", name=team,
                       marker=dict(size=10, color="red" if team == "Team 1" else "blue"),
                       text=team_df["Player ID"]),
            row=2, col=2
        )
    
    # Format dashboard
    fig.update_layout(
        title_text="Player Performance Report",
        height=800,
        width=1200
    )
    
    # Save dashboard
    fig.write_html(output_path)
    
    print(f"Created interactive dashboard at: {output_path}")

def generate_player_performance_report(image_paths):
    """
    Create a comprehensive report on player performance.
    
    Parameters:
        image_paths: List of image paths.
    """
    print("\nGenerating comprehensive player performance report...")
    
    # Detect players in images
    all_players = []
    for i, image_path in enumerate(image_paths):
        players = simulate_player_detection(image_path)
        all_players.append(players)
        
        # Create illustrative image
        output_path = os.path.join(OUTPUT_DIR, f"player_detection_{i+1}.jpg")
        visualize_player_detection(image_path, players, output_path)
    
    # Track player movements
    tracks = track_player_movements(image_paths)
    
    # Calculate player statistics
    player_stats = calculate_player_statistics(tracks)
    
    # Create heatmap
    heatmap_path = os.path.join(OUTPUT_DIR, "player_heatmap.jpg")
    create_heatmap(tracks, image_paths[0], heatmap_path)
    
    # Create player performance chart
    chart_path = os.path.join(OUTPUT_DIR, "player_performance_chart.jpg")
    create_player_performance_chart(player_stats, chart_path)
    
    # Create team comparison chart
    team_chart_path = os.path.join(OUTPUT_DIR, "team_comparison_chart.jpg")
    create_team_comparison_chart(player_stats, team_chart_path)
    
    # Create interactive dashboard
    dashboard_path = os.path.join(OUTPUT_DIR, "interactive_dashboard.html")
    create_interactive_dashboard(player_stats, tracks, dashboard_path)
    
    print("\nAll report files have been created in the folder:", OUTPUT_DIR)
    print("You can open the HTML file for the interactive dashboard to view the detailed report.")

if __name__ == "__main__":
    # Download images
    images = download_sample_images()
    
    if images:
        generate_player_performance_report(images)
    else:
        print("No images were downloaded. Please check your internet connection or links.")

##------------------------------------------------------------------------------------------
# from datasets import load_dataset
# import os
# import cv2
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import requests

# # Configure paths
# OUTPUT_DIR = "player_analysis_output"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def download_football_images():
#     """
#     Download sample football images from Hugging Face datasets for analysis.
#     """
#     print("Downloading football images from Hugging Face datasets...")

#     # Load datasets
#     dataset_1 = load_dataset("Voxel51/Football-Player-Segmentation", split="train")
#     dataset_2 = load_dataset("Voxel51/SoccerNet-V3", split="train")
    
#     # You can specify an amount of images to download (first 5 images from each dataset)
#     images = []
#     for i, dataset in enumerate([dataset_1, dataset_2]):
#         for img_data in dataset[:5]:  # Modify the number as per your need
#             img_url = img_data['image_url']  # Assuming the dataset has an image_url field
#             try:
#                 # Download image from the dataset URL
#                 response = requests.get(img_url)
#                 img = Image.open(BytesIO(response.content))
#                 img_path = os.path.join(OUTPUT_DIR, f"sample_image_{i+1}_{img_data['id']}.jpg")
#                 img.save(img_path)
#                 images.append(img_path)
#                 print(f"Downloaded image from {img_url} and saved to {img_path}")
#             except Exception as e:
#                 print(f"Failed to download image from {img_url}: {str(e)}")
    
#     return images

# def simulate_player_detection(image_path):
#     """
#     Simulate player detection in an image.
    
#     Parameters:
#         image_path: Path to the image.
        
#     Returns:
#         List of detected players' information (locations and IDs).
#     """
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Failed to read image from {image_path}")
#         return []
    
#     height, width = img.shape[:2]
    
#     # Simulate player detection (in a real application, a YOLO model would be used here)
#     num_players = np.random.randint(5, 15)  # Random number of players
    
#     players = []
#     for i in range(num_players):
#         # Create a random bounding box
#         x1 = np.random.randint(0, width - 100)
#         y1 = np.random.randint(0, height - 200)
#         w = np.random.randint(50, 100)
#         h = np.random.randint(100, 200)
#         x2 = x1 + w
#         y2 = y1 + h
        
#         # Assign a random team ID (0 or 1)
#         team_id = np.random.randint(0, 2)
        
#         # Create player information
#         player = {
#             "id": i + 1,
#             "bbox": [x1, y1, x2, y2],
#             "team_id": team_id,
#             "confidence": np.random.uniform(0.7, 0.99)
#         }
        
#         players.append(player)
    
#     return players

# def visualize_player_detection(image_path, players, output_path):
#     """
#     Create an image showing player locations.
    
#     Parameters:
#         image_path: Path to the image.
#         players: Players' information.
#         output_path: Path to save the image.
#     """
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Failed to read image from {image_path}")
#         return
    
#     # Copy of the image for drawing
#     img_draw = img.copy()
    
#     # Team colors
#     team_colors = [(0, 0, 255), (255, 0, 0)]  # Red and Blue
    
#     # Draw rectangles around players
#     for player in players:
#         bbox = player["bbox"]
#         team_id = player["team_id"]
#         player_id = player["id"]
#         confidence = player["confidence"]
        
#         x1, y1, x2, y2 = bbox
        
#         # Draw rectangle
#         cv2.rectangle(img_draw, (x1, y1), (x2, y2), team_colors[team_id], 2)
        
#         # Add player ID
#         label = f"Player {player_id} ({confidence:.2f})"
#         cv2.putText(img_draw, label, (x1, y1-10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_colors[team_id], 2)
    
#     # Save the image
#     cv2.imwrite(output_path, img_draw)
    
#     print(f"Created player detection image at: {output_path}")
#     print(f"Number of detected players: {len(players)}")

# def track_player_movements(image_paths):
#     """
#     Simulate tracking player movements across a series of images.
    
#     Parameters:
#         image_paths: List of image paths.
        
#     Returns:
#         Player tracking data.
#     """
#     print("Simulating player movement tracking...")
    
#     all_tracks = {}
    
#     for frame_idx, image_path in enumerate(image_paths):
#         # Detect players in the current frame
#         players = simulate_player_detection(image_path)
        
#         # Update player tracks
#         for player in players:
#             player_id = player["id"]
            
#             if player_id not in all_tracks:
#                 all_tracks[player_id] = {
#                     "frames": [],
#                     "positions": [],
#                     "team_id": player["team_id"]
#                 }
            
#             # Extract player center position
#             bbox = player["bbox"]
#             center_x = (bbox[0] + bbox[2]) / 2
#             center_y = (bbox[1] + bbox[3]) / 2
            
#             # Add frame and position information
#             all_tracks[player_id]["frames"].append(frame_idx)
#             all_tracks[player_id]["positions"].append((center_x, center_y))
    
#     return all_tracks

# def calculate_player_statistics(tracks):
#     """
#     Calculate player performance statistics from tracking data.
    
#     Parameters:
#         tracks: Player tracking data.
        
#     Returns:
#         Player statistics.
#     """
#     print("Calculating player performance statistics...")
    
#     player_stats = {}
    
#     for player_id, track_data in tracks.items():
#         positions = track_data["positions"]
#         team_id = track_data["team_id"]
        
#         # Calculate total distance traveled
#         total_distance = 0
#         for i in range(1, len(positions)):
#             x1, y1 = positions[i-1]
#             x2, y2 = positions[i]
#             distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#             total_distance += distance
        
#         # Calculate average speed (distance per frame)
#         avg_speed = total_distance / len(positions) if len(positions) > 0 else 0
        
#         # Calculate average position (mean location)
#         avg_x = np.mean([p[0] for p in positions]) if positions else 0
#         avg_y = np.mean([p[1] for p in positions]) if positions else 0
        
#         # Calculate coverage area (standard deviation of positions)
#         std_x = np.std([p[0] for p in positions]) if len(positions) > 1 else 0
#         std_y = np.std([p[1] for p in positions]) if len(positions) > 1 else 0
#         coverage_area = std_x * std_y
        
#         # Store statistics
#         player_stats[player_id] = {
#             "team_id": team_id,
#             "total_distance": total_distance,
#             "avg_speed": avg_speed,
#             "avg_position": (avg_x, avg_y),
#             "coverage_area": coverage_area
#         }
    
#     return player_stats

# def create_heatmap(tracks, image_path, output_path):
#     """
#     Create a heatmap of player locations.
    
#     Parameters:
#         tracks: Player tracking data.
#         image_path: Path to the original image.
#         output_path: Path to save the heatmap.
#     """
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Failed to read image from {image_path}")
#         return
    
#     height, width = img.shape[:2]
    
#     # Create a matrix to store density data
#     heatmap = np.zeros((height, width), dtype=np.float32)
    
#     # Add points for each player location
#     for player_id, track_data in tracks.items():
#         positions = track_data["positions"]
        
#         for pos in positions:
#             x, y = int(pos[0]), int(pos[1])
            
#             # Ensure coordinates are within image bounds
#             if 0 <= x < width and 0 <= y < height:
#                 # Add a heat point with Gaussian effect
#                 sigma = 30  # Spread of the effect
#                 for i in range(max(0, y-sigma*3), min(height, y+sigma*3)):
#                     for j in range(max(0, x-sigma*3), min(width, x+sigma*3)):
#                         dist = np.sqrt((i-y)**2 + (j-x)**2)
#                         influence = np.exp(-(dist**2) / (2*sigma**2))
#                         heatmap[i, j] += influence
    
#     # Normalize the heatmap
#     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
#     heatmap = heatmap.astype(np.uint8)
    
#     # Apply color map
#     heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
#     # Overlay heatmap on original image
#     alpha = 0.6
#     overlay = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    
#     # Save the image
#     cv2.imwrite(output_path, overlay)
    
#     print(f"Created heatmap at: {output_path}")

# def create_player_performance_chart(player_stats, output_path):
#     """
#     Create a chart for player performance.
    
#     Parameters:
#         player_stats: Player statistics.
#         output_path: Path to save the chart.
#     """
#     # Convert data to DataFrame
#     data = []
#     for player_id, stats in player_stats.items():
#         data.append({
#             "Player ID": f"Player {player_id}",
#             "Team": f"Team {stats['team_id'] + 1}",
#             "Distance": stats["total_distance"],
#             "Speed": stats["avg_speed"],
#             "Coverage": stats["coverage_area"]
#         })
    
#     df = pd.DataFrame(data)
    
#     # Create multi-chart
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # Distance bar plot
#     sns.barplot(x="Player ID", y="Distance", hue="Team", data=df, ax=axes[0])
#     axes[0].set_title("Distance Covered by Each Player")
#     axes[0].set_ylabel("Distance")
#     axes[0].set_xlabel("Player ID")
    
#     # Speed bar plot
#     sns.barplot(x="Player ID", y="Speed", hue="Team", data=df, ax=axes[1])
#     axes[1].set_title("Average Speed of Each Player")
#     axes[1].set_ylabel("Speed")
#     axes[1].set_xlabel("Player ID")
    
#     # Coverage area bar plot
#     sns.barplot(x="Player ID", y="Coverage", hue="Team", data=df, ax=axes[2])
#     axes[2].set_title("Coverage Area of Each Player")
#     axes[2].set_ylabel("Coverage Area")
#     axes[2].set_xlabel("Player ID")
    
#     # Format the chart
#     plt.tight_layout()
    
#     # Save chart
#     plt.savefig(output_path)
    
#     print(f"Created player performance chart at: {output_path}")

# def create_team_comparison_chart(player_stats, output_path):
#     """
#     Create a chart comparing the performance of two teams.
    
#     Parameters:
#         player_stats: Player statistics.
#         output_path: Path to save the chart.
#     """
#     # Aggregate team statistics
#     team_stats = {0: {"distance": 0, "speed": 0, "coverage": 0, "count": 0},
#                    1: {"distance": 0, "speed": 0, "coverage": 0, "count": 0}}
    
#     for player_id, stats in player_stats.items():
#         team_id = stats["team_id"]
#         team_stats[team_id]["distance"] += stats["total_distance"]
#         team_stats[team_id]["speed"] += stats["avg_speed"]
#         team_stats[team_id]["coverage"] += stats["coverage_area"]
#         team_stats[team_id]["count"] += 1
    
#     # Calculate averages
#     for team_id in team_stats:
#         if team_stats[team_id]["count"] > 0:
#             team_stats[team_id]["avg_distance"] = team_stats[team_id]["distance"] / team_stats[team_id]["count"]
#             team_stats[team_id]["avg_speed"] = team_stats[team_id]["speed"] / team_stats[team_id]["count"]
#             team_stats[team_id]["avg_coverage"] = team_stats[team_id]["coverage"] / team_stats[team_id]["count"]
    
#     # Create DataFrame for the chart
#     data = []
#     for team_id, stats in team_stats.items():
#         if stats["count"] > 0:
#             data.append({
#                 "Team": f"Team {team_id + 1}",
#                 "Metric": "Distance",
#                 "Value": stats["avg_distance"]
#             })
#             data.append({
#                 "Team": f"Team {team_id + 1}",
#                 "Metric": "Speed",
#                 "Value": stats["avg_speed"]
#             })
#             data.append({
#                 "Team": f"Team {team_id + 1}",
#                 "Metric": "Coverage",
#                 "Value": stats["avg_coverage"]
#             })
    
#     df = pd.DataFrame(data)
    
#     # Create a comparison chart
#     plt.figure(figsize=(12, 8))
    
#     # Comparison bar plot
#     sns.barplot(x="Metric", y="Value", hue="Team", data=df)
    
#     # Format the chart
#     plt.title("Comparison of Team Performance")
#     plt.ylabel("Value")
#     plt.xlabel("Metric")
    
#     # Save chart
#     plt.savefig(output_path)
    
#     print(f"Created team comparison chart at: {output_path}")

# def create_interactive_dashboard(player_stats, tracks, output_path):
#     """
#     Create an interactive dashboard for player performance.
    
#     Parameters:
#         player_stats: Player statistics.
#         tracks: Player tracking data.
#         output_path: Path to save the dashboard.
#     """
#     # Convert data to DataFrame
#     data = []
#     for player_id, stats in player_stats.items():
#         data.append({
#             "Player ID": f"Player {player_id}",
#             "Team": f"Team {stats['team_id'] + 1}",
#             "Distance": stats["total_distance"],
#             "Speed": stats["avg_speed"],
#             "Coverage": stats["coverage_area"],
#             "X Position": stats["avg_position"][0],
#             "Y Position": stats["avg_position"][1]
#         })
    
#     df = pd.DataFrame(data)
    
#     # Create interactive dashboard
#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=("Distance Covered", "Average Speed", "Coverage Area", "Player Locations"),
#         specs=[[{"type": "bar"}, {"type": "bar"}],
#                [{"type": "bar"}, {"type": "scatter"}]]
#     )
    
#     # Add distance bar plot
#     fig.add_trace(
#         go.Bar(x=df["Player ID"], y=df["Distance"], name="Distance", 
#                marker_color=df["Team"].map({"Team 1": "red", "Team 2": "blue"})),
#         row=1, col=1
#     )
    
#     # Add speed bar plot
#     fig.add_trace(
#         go.Bar(x=df["Player ID"], y=df["Speed"], name="Speed",
#                marker_color=df["Team"].map({"Team 1": "red", "Team 2": "blue"})),
#         row=1, col=2
#     )
    
#     # Add coverage area bar plot
#     fig.add_trace(
#         go.Bar(x=df["Player ID"], y=df["Coverage"], name="Coverage",
#                marker_color=df["Team"].map({"Team 1": "red", "Team 2": "blue"})),
#         row=2, col=1
#     )
    
#     # Add player locations scatter plot
#     for team in ["Team 1", "Team 2"]:
#         team_df = df[df["Team"] == team]
#         fig.add_trace(
#             go.Scatter(x=team_df["X Position"], y=team_df["Y Position"], 
#                        mode="markers", name=team,
#                        marker=dict(size=10, color="red" if team == "Team 1" else "blue"),
#                        text=team_df["Player ID"]),
#             row=2, col=2
#         )
    
#     # Format dashboard
#     fig.update_layout(
#         title_text="Player Performance Report",
#         height=800,
#         width=1200
#     )
    
#     # Save dashboard
#     fig.write_html(output_path)
    
#     print(f"Created interactive dashboard at: {output_path}")

# def generate_player_performance_report(image_paths):
#     """
#     Create a comprehensive report on player performance.
    
#     Parameters:
#         image_paths: List of image paths.
#     """
#     print("\nGenerating comprehensive player performance report...")
    
#     # Detect players in images
#     all_players = []
#     for i, image_path in enumerate(image_paths):
#         players = simulate_player_detection(image_path)
#         all_players.append(players)
        
#         # Create illustrative image
#         output_path = os.path.join(OUTPUT_DIR, f"player_detection_{i+1}.jpg")
#         visualize_player_detection(image_path, players, output_path)
    
#     # Track player movements
#     tracks = track_player_movements(image_paths)
    
#     # Calculate player statistics
#     player_stats = calculate_player_statistics(tracks)
    
#     # Create heatmap
#     heatmap_path = os.path.join(OUTPUT_DIR, "player_heatmap.jpg")
#     create_heatmap(tracks, image_paths[0], heatmap_path)
    
#     # Create player performance chart
#     chart_path = os.path.join(OUTPUT_DIR, "player_performance_chart.jpg")
#     create_player_performance_chart(player_stats, chart_path)
    
#     # Create team comparison chart
#     team_chart_path = os.path.join(OUTPUT_DIR, "team_comparison_chart.jpg")
#     create_team_comparison_chart(player_stats, team_chart_path)
    
#     # Create interactive dashboard
#     dashboard_path = os.path.join(OUTPUT_DIR, "interactive_dashboard.html")
#     create_interactive_dashboard(player_stats, tracks, dashboard_path)
    
#     print("\nAll report files have been created in the folder:", OUTPUT_DIR)
#     print("You can open the HTML file for the interactive dashboard to view the detailed report.")

# # Main code to use the images from Hugging Face datasets
# if __name__ == "__main__":
#     images = download_football_images()

#     if images:
#         generate_player_performance_report(images)
#     else:
#         print("No images were downloaded. Please check your internet connection or links.")