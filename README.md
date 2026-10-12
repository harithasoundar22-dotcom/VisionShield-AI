# VisionShield AI — HackZen 2.0

A beginner-friendly surface-inspection prototype featuring:
- OpenCV image-quality checks (blur and brightness)
- Edge visualization as a visual aid
- Optional HOG + SVM classifier trained on labelled normal/defect images
- Basic validation metrics during training

## Important limitations
This is a prototype, not a certified industrial inspection tool. Edge visualization does not confirm defects. The classifier will not work until you train it with suitable labelled images. Performance depends on the dataset and must be tested honestly.

## 1. Setup
Use Python 3.10 or 3.11 if possible for easier package compatibility.

Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, try:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Run the app (quality checks work without a trained model)
```powershell
streamlit run app.py
```

## 3. Add labelled images to train the classifier
Place images in:
- `data/train/normal/`
- `data/train/defect/`

Use images of the same type of surface and label them correctly. Aim for dozens or hundreds of images per class if possible. Do not use the same images in both training and test folders.

Train:
```powershell
python train_model.py
```

Then restart the app:
```powershell
streamlit run app.py
```

## Team split
- Member 1: find, check, and label dataset; run training and explain HOG/SVM.
- Member 2: understand image-quality checks and OpenCV edge visualization.
- Member 3: run Streamlit app, test cases, README, demo video, GitHub integration.

## Demo outline
1. Upload a clear image.
2. Upload a blurry/dark image and show the quality warning.
3. If trained, show a normal example and a defect example.
4. Show validation metrics from `train_model.py`.
5. Explain limitations and next steps.
