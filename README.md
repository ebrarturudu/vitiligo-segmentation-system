\# Analysis and Evaluation of Vitiligo Severity Using Semantic Segmentation in Clinical Dermatology



This project is a Flask-based web application for vitiligo lesion segmentation

using a SAM + U-Net hybrid deep learning model.  

The system focuses on \*\*model inference, web serving, and deployability\*\* rather than model training.



---

\## Project Overview

This project presents a deep learning–based vitiligo segmentation system that combines the Segment Anything Model (SAM) with a U-Net architecture to identify and quantify vitiligo-affected regions in skin images.

The system includes an end-to-end inference pipeline and a lightweight Flask-based web interface for visualization and analysis.

---



\## Key Features



\- Image upload and vitiligo lesion segmentation

\- Lesion area ratio calculation

\- Web-based visualization with Flask

\- Batch inference support

\- Inference latency measurement

\- Dockerized for portable deployment



---



\## System Architecture

\-Backbone: U-Net

\-Encoder guidance: Segment Anything Model (SAM)

\-Output: Binary segmentation mask

\-Post-processing: Lesion area ratio calculation

Training notebooks and pretrained weights are excluded due to academic publication constraints.


---

\## Dataset

The dataset consists of dermatological images collected from multiple academic and open-access sources.

Due to ethical considerations, patient privacy, and ongoing academic publication, the dataset is not publicly available.

Researchers interested in academic collaboration may contact the author.


\## Project Structure



VITILIGO\_APP/

│

├─ app.py

├─ batch\_send\_analiz.py

├─ batch\_send\_predict.py

├─ weblatency.py

│

├─ model/

│ ├─ sam\_unet.py

│ ├─ vitiligosegmenter.py

│ ├─ segment.py

│ ├─ segment\_anything/

│ └─ models/

│ └─ README.txt

│

├─ static/

├─ templates/

├─ test\_images/

│

├─ Dockerfile

├─ .dockerignore

├─ .gitignore

├─ requirements.txt

├─ latency\_log.csv

└─ README.md





---



\## Run with Docker



\### Build image

```bash

docker build -t vitiligo-app .



Run container

docker run -p 8080:8080 vitiligo-app



Then open:

http://localhost:8080



Notes



* Model weights (.pth) are not included due to ongoing academic publication.
* The application automatically falls back to CPU if CUDA is not available.
* latency\_log.csv contains sample inference latency measurements.



Technologies Used



* Python
* Flask
* PyTorch
* Docker
* HTML / CSS (Jinja Templates)





Author



Ebrar Türüdü

GitHub: https://github.com/ebrarturudu







