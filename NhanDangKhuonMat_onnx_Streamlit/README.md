# Nhận diện khuôn mặt ONNX Streamlit

## Bước 1: 
- Cài thư viện: 
```bash
pip install numpy==1.24.4 opencv-contrib-python==4.8.0.76
pip install streamlit Pillow
```

> Lưu ý: Không dùng opencv-python, vì nó không chứa các API như cv.FaceDetectorYN hoặc cv.FaceRecognizerSF. Những API này nằm trong opencv-contrib-python.

- TẢI MODELS (.onnx)
https://utexlms.hcmute.edu.vn/mod/resource/view.php?id=461929

## Bước 2:

- Cài thư viện:
```bash
pip install matplotlib==3.7.1 scikit-learn==1.2.2 joblib==1.2.0
```