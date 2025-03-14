# 📡 Anomaly Detection in IoT Devices using Autoencoder Networks

## 📌 Project Overview

This project presents an advanced **anomaly detection system for IoT devices**, leveraging deep learning techniques, specifically **autoencoder neural networks**. The system is designed to detect abnormal behaviors in IoT networks by analyzing patterns in collected data, ensuring higher security and reliability in smart environments.

The research focuses on **unsupervised learning** to identify anomalies without requiring labeled datasets, making it highly adaptable to real-world IoT scenarios. The key advantage of our approach is that the model is trained **exclusively on benign data**, making it highly flexible and capable of detecting previously unseen attacks. By learning the normal behavior of IoT devices, the system can flag any deviation as a potential threat, ensuring robust security even against novel attack strategies.

## ✨ Key Features

✅ **Autoencoder-based anomaly detection** 🤖  
✅ **Unsupervised learning for robust adaptability** 🔍  
✅ **Optimized for IoT environments** 📡  
✅ **Efficient data processing with high detection accuracy** 📊  
✅ **Scalable and deployable in real-world applications** 🚀  

## 🧠 How the Autoencoder Works

An **autoencoder** is a type of neural network designed to learn efficient representations of input data. It consists of two main components:

- **Encoder:** Compresses input data into a lower-dimensional feature space.
- **Bottleneck Layer:** Stores the compressed latent representation of the input.
- **Decoder:** Reconstructs the original input from the compressed data.
- **Anomaly Detection Mechanism:** Since the network is trained on normal IoT traffic, it reconstructs normal data with minimal error. Anomalies cause higher reconstruction errors, allowing detection.

## 🔥 Types of Attacks Detected

The system is specifically designed to detect **Mirai and BASHLITE botnet attacks**, which target IoT devices to integrate them into large-scale botnets used for malicious purposes. The primary types of attacks detected include:

- **Command & Control (C2) Communication:** Identifying unusual outbound traffic patterns where compromised devices communicate with malicious servers.
- **Denial of Service (DoS) Attacks:** Detecting high-volume request spikes attempting to overwhelm a target.
- **Brute Force Attacks:** Recognizing unauthorized login attempts targeting weakly secured IoT devices.
- **Malware Injection:** Identifying attempts to download and execute malicious payloads.
- **Data Exfiltration:** Detecting unauthorized data transmissions from compromised devices.

## 📡 IoT Devices Analyzed

The dataset used for training and evaluation includes network traffic from various IoT devices commonly found in smart home environments. Below is a summary of the devices considered:

| Device Type       | Model Considered | Number of Benign Instances |
|------------------|-----------------|---------------------------|
| **Doorbell**  | Daanmi | 45,548 |
| **Doorbell**  | Ennio | 39,100 |
| **Thermostat** | Ecobee | 13,113 |
| **Baby Monitor** | Philips B120N/10 | 175,240 |
| **Security Camera** | Provision PT-737E | 62,154 |
| **Security Camera** | Provision PT-838 | 86,514 |
| **Security Camera** | SimpleHome XCS7-1002-WHT | 46,528 |
| **Security Camera** | SimpleHome XCS7-1003-WHT | 19,828 |
| **Webcam** | Samsung SNH 1011 N | 52,150 |

These devices were selected to represent a broad range of IoT applications, ensuring the model generalizes well across different network environments.

## 📂 Dataset and Data Type

The dataset consists of network traffic data collected from IoT devices, focusing primarily on **UDP packets exchanged by the devices**. The main characteristics of the dataset include:

- **Packet-Level Features:** Includes attributes such as packet size, source and destination IP addresses, port numbers, timestamp, and protocol type.
- **Traffic Flow Analysis:** Aggregated statistics on communication patterns between IoT devices and external entities.
- **Time-Series Data:** Sequences of packet exchanges over time to identify behavioral patterns.
- **Anomaly Labels:** The dataset contains normal and attack-labeled instances to evaluate detection performance.

Importantly, our model is trained **only on benign data** to ensure maximum adaptability to emerging threats. Unlike traditional systems that rely on attack signatures, our approach enables real-time detection of **zero-day attacks** and other novel threats without requiring prior knowledge of their characteristics.

## 📊 Results and Performance

The model was evaluated using real-world IoT traffic datasets, demonstrating **high effectiveness in anomaly detection**. Key results include:

- **Detection Accuracy:** High accuracy in identifying malicious activities.
- **False Positive Rate (FPR):** Low FPR ensuring minimal misclassification of normal traffic.
- **Latency:** Optimized for near real-time detection.
- **Scalability:** Successfully tested on datasets containing traffic from multiple IoT devices.
- **Mirai & BASHLITE Attack Detection Rate:** Effective identification of botnet-infected devices.

These results validate the system's ability to identify security threats with high precision while maintaining a low false positive rate, making it a practical solution for real-world IoT security applications.

## 📜 License

This project is released under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to submit pull requests or report issues.

🚀 **Enhancing IoT security with deep learning-powered anomaly detection!**
