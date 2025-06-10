# Smart-Traffic-Analysis-and-Management 
This repository contains a complete implementation of **Smart City Traffic Analysis and Management**, where real-time traffic data is ingested, predictions are made for traffic volumes, traffic light schedules are optimized, and congestion hotspots are detected. The backend is implemented using **Django** and **Django REST Framework** to serve APIs, while the frontend is powered by **HTML** and **Bootstrap**. Power BI can be used for real-time data visualization.

## Features

- **Real-Time Traffic Data Ingestion**: Simulated data ingestion with chunk processing.
- **Traffic Volume Prediction**: Uses **Random Forest** to predict traffic volumes.
- **Traffic Light Optimization**: Uses **Google OR-Tools** for traffic light schedule optimization.
- **Congestion Hotspot Detection**: Uses **K-Means clustering** to identify congestion hotspots.
- **Plot Generation**: Generates traffic volume distribution, daily average traffic volume, and feature correlation matrix using **Matplotlib** and **Seaborn**.
- **Simple Frontend UI**: Displays predictions, optimized traffic light schedules, and visualizations using **HTML** and **Bootstrap**.
- **Power BI Integration**: Connects the Django API to **Power BI** for real-time dashboards.

## Getting Started

These instructions will help you set up the project locally for development and testing purposes.

### Prerequisites

- **Python** 3.x
- **Django** 3.x or higher
- **Django REST Framework** for creating API endpoints
- **Power BI Desktop** (for visualization)
- **Matplotlib** and **Seaborn** (for plot generation)

### Installing Dependencies


1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/smart_city_traffic_analysis.git
   cd smart_city_traffic_analysis
2. Create a virtual environment and activate it:
   ```bash
     python -m venv venv
     source venv/bin/activate  # For Linux/Mac
     venv\Scripts\activate     # For Windows
3. 
