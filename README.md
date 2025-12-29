# SynergyX – 5G mmWave Throughput Prediction & Optimization

SynergyX is an AI-powered solution built during our final year at **Esprit School of Engineering** to predict and optimize **5G mmWave network throughput**. It combines a hybrid **CNN + LSTM** model (leveraging raw and smoothed throughput signals), an **MLOps pipeline** for reproducible training and deployment, and a **real-time dashboard** to simulate user mobility and network performance.

This initiative aligns with **SDG 9 (Industry, Innovation & Infrastructure)**, helping telecom operators improve capacity planning, stability, and user experience with a scalable, data-driven approach.

## Demo
- 🎥 Video walkthrough: https://youtu.be/SUA1eYoUz34?si=DLL-2I5aSgyyzBw1

## Highlights
- **Hybrid CNN + LSTM** model combining raw and smoothed throughput.
- **MLOps pipeline** with MLflow for experiment tracking and deployment.
- **Real-time visualization dashboard** for throughput comparison, mobility simulation, and coverage insights.

## Project Architecture
![1747728150175](https://github.com/user-attachments/assets/c325bfc2-fc4e-45e8-b8a7-a4b4b113f0fd)


## Dashboard Preview
![1747728150219](https://github.com/user-attachments/assets/ca966b71-5fb4-4f40-9e6c-f014f5450c2f)


## Tech Stack
**Data Handling**: pandas, NumPy, scikit-learn  
**Modeling**: Keras, TensorFlow  
**Deployment**: FastAPI, Docker  
**Visualization**: Django, Tailwind CSS, Google Maps

## Quickstart (Docker)
> Requires Docker + Docker Compose.

```bash
make build-mlflow
make build-backend
make build-frontend
make run-container
```

### Services
- **Frontend Dashboard**: http://localhost/ (port 80)
- **Backend API**: http://localhost:8000
- **MLflow Tracking**: http://localhost:5000
- **Elasticsearch**: http://localhost:9200
- **Kibana**: http://localhost:5601

## Acknowledgements
Grateful to our instructors **Rahma Bouraoui**, **Safa Cherif**, and **Zaineb Labidi** for their continuous support.

## Team & Project Context
SynergyX was developed as part of the **Integrated Project** at **ESPRIT (École Supérieure Privée d'Ingénierie et de Technologies)**.

---
How do you see AI transforming telecom infrastructures in the near future? We’d love to hear your thoughts and experiences.
