# VisionShield AI — Project Documentation

## Problem
Visual inspection of manufactured surfaces can be repetitive. Poorly lit or blurry images can make inspection less reliable.

## Proposed solution
VisionShield AI is a prototype that:
1. checks simple image-quality indicators;
2. displays edge information as a visual aid;
3. optionally classifies an image as `normal` or `defect` using a HOG + SVM model trained on labelled images.

## Methods
- OpenCV grayscale conversion and Laplacian variance for a basic blur indicator.
- Mean grayscale intensity as a simple brightness indicator.
- Canny edge detection for visual explanation.
- HOG (Histogram of Oriented Gradients) features and a Support Vector Machine for optional binary classification.

## Innovation claim
The project combines image-quality warnings with a defect-classification workflow and human verification. This is a proposed prototype combination, not a claim of being the first system of its kind. Any claim of improved performance must be supported by measured experiments.

## Evaluation plan
- Collect separate labelled normal and defect images.
- Keep training and testing images separate.
- Record precision, recall, F1-score and confusion matrix.
- Test clear, blurry, dark, and bright images.
- Report errors and limitations; do not report made-up accuracy.

## Limitations
- HOG + SVM can perform poorly if the dataset is small or differs from real images.
- Image quality thresholds are simple heuristics and may need calibration.
- Edge maps show boundaries and texture, not verified defects.
- A normal/defect image classifier does not necessarily localize defects.
- Not suitable for safety-critical decisions without extensive validation.

## Future work
- Use a larger, well-labelled industrial dataset.
- Compare HOG + SVM with a pretrained CNN or anomaly-detection model.
- Add defect localization/segmentation with appropriate labels.
- Improve calibration and evaluate robustness under different lighting and viewpoints.
