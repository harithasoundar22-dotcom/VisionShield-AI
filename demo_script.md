# 2–3 minute demo script

**Opening (15 sec):**
"Hello, we are presenting VisionShield AI, a computer-vision prototype for surface inspection. It checks image quality and supports defect classification."

**Upload and quality checks (35 sec):**
"First, we upload an image. The system measures a simple blur score and average brightness. If an image appears blurry or too dark, it warns the user that the inspection may be unreliable."

**Visual aid (25 sec):**
"The edge visualization makes surface structure easier to inspect. Edges are only a visual aid; they are not treated as proof of a defect."

**Classifier (35 sec, only if trained):**
"We trained a HOG + SVM classifier on labelled normal and defect images. Here are example predictions. These results depend on the quality and variety of our dataset."

**Evaluation (30 sec, only with real results):**
"We evaluated on a separate validation set and report the precision, recall, F1-score, and confusion matrix. We do not claim performance beyond what our tests show."

**Limitations and close (20 sec):**
"This is a prototype, not a certified industrial inspection system. Future work includes more data, improved localization, and comparison with deep-learning models. Thank you."
