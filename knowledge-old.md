# IoMT Security Classification System

## Project Overview
This project implements a machine learning system for detecting and classifying attacks in Internet of Medical Things (IoMT) network traffic. The system analyzes both WiFi/MQTT and Bluetooth traffic to identify various types of attacks.

## Key Components

### Data Types
- WiFi/MQTT traffic in CSV format
- Bluetooth traffic in PCAP format

### Attack Types
- DoS (Denial of Service)
- DDoS (Distributed Denial of Service)
- MQTT Malformed Data
- ARP Spoofing
- Reconnaissance
- Benign Traffic (Normal)

### System Architecture
- DataLoader: Handles efficient loading and preprocessing of network traffic data
- FeatureProcessor: Manages feature selection and processing
- ModelTrainer: Implements multiple ML models for classification
- Evaluator: Provides comprehensive evaluation metrics and visualizations

### Machine Learning Models
- Random Forest
- Logistic Regression
- AdaBoost
- Neural Network (MLP)

## Development Guidelines
- Use batch processing for large datasets
- Implement memory-efficient data handling
- Follow proper logging practices
- Save comprehensive evaluation results

## Project Structure
```
.
├── Dataset/
│   ├── Bluetooth/
│   │   └── attacks/
│   └── WiFI_and_MQTT/
│       └── attacks/
├── c-enh-old-models-less-performance.py
└── evaluation_results/
```

## Verification Protocol
After making changes:
1. Run type checking: `mypy c-enh.py`
2. Run tests if available
3. Verify memory usage for large datasets

## Dependencies
See requirements.txt for specific package versions