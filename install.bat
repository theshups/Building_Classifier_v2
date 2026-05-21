@echo off
chcp 65001 >nul
echo ============================================================
echo   BuildingYOLO  -  Dependency Installer
echo   ResNet50V2 classification + YOLOv8n detection
echo   Installs in correct order to avoid numpy conflicts
echo ============================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    echo Install Python 3.10 from https://python.org
    pause
    exit /b 1
)

python --version

echo.
echo [1/10] NumPy 1.26.4  (must be first - TF requires numpy 1.x)
pip install "numpy==1.26.4" --force-reinstall --quiet
if %errorlevel% neq 0 ( echo FAILED: numpy & pause & exit /b 1 )

echo [2/10] TensorFlow 2.15.0
pip install tensorflow==2.15.0 --quiet
if %errorlevel% neq 0 ( echo FAILED: tensorflow & pause & exit /b 1 )

echo [3/10] h5py + Pillow
pip install "h5py==3.11.0" "pillow==10.3.0" --quiet
if %errorlevel% neq 0 ( echo FAILED: h5py or pillow & pause & exit /b 1 )

echo [4/10] Data science tools
pip install "scikit-learn==1.4.2" "matplotlib==3.8.4" "pandas==2.2.2" "tqdm==4.66.2" --quiet
if %errorlevel% neq 0 ( echo FAILED: data tools & pause & exit /b 1 )

echo [5/10] FastAPI + Uvicorn
pip install "fastapi==0.110.0" "uvicorn[standard]==0.29.0" "python-multipart==0.0.9" "aiofiles==23.2.1" --quiet
if %errorlevel% neq 0 ( echo FAILED: fastapi & pause & exit /b 1 )

echo [6/10] OpenCV 4.9.0.80  (numpy-compatible, must be before ultralytics)
pip install "opencv-python==4.9.0.80" --quiet
if %errorlevel% neq 0 ( echo FAILED: opencv & pause & exit /b 1 )

echo [7/10] YOLOv8  (ultralytics)
pip install "ultralytics==8.2.0" --quiet
if %errorlevel% neq 0 ( echo FAILED: ultralytics & pause & exit /b 1 )

echo [8/10] Roboflow  (pipeline dataset download)
pip install roboflow --quiet
if %errorlevel% neq 0 ( echo FAILED: roboflow & pause & exit /b 1 )

echo [9/10] Re-pinning NumPy 1.26.4  (ultralytics and roboflow may upgrade it)
pip install "numpy==1.26.4" --force-reinstall --quiet
if %errorlevel% neq 0 ( echo FAILED: numpy re-pin & pause & exit /b 1 )

echo [10/10] Re-pinning OpenCV 4.9.0.80
pip install "opencv-python==4.9.0.80" --force-reinstall --quiet
if %errorlevel% neq 0 ( echo FAILED: opencv re-pin & pause & exit /b 1 )

echo.
echo Removing conflicting packages...
pip uninstall opencv-python-headless -y >nul 2>&1

echo.
echo ============================================================
echo   Verifying installations
echo ============================================================
python -c "import numpy; v=numpy.__version__; s='OK' if v.startswith('1.') else 'WRONG VERSION'; print('  numpy       :', v, s)"
python -c "import tensorflow as tf; print('  tensorflow  :', tf.__version__)"
python -c "import cv2; print('  opencv      :', cv2.__version__)"
python -c "import ultralytics; print('  ultralytics : OK')"
python -c "import roboflow; print('  roboflow    : OK')"
python -c "import fastapi; print('  fastapi     :', fastapi.__version__)"
python -c "import uvicorn; print('  uvicorn     :', uvicorn.__version__)"
echo.

echo ============================================================
echo   All done!
echo.
echo   NEXT STEPS:
echo   1. Run setup_roboflow.bat  to save your Roboflow API key
echo   2. Close and reopen PowerShell after setup_roboflow.bat
echo   3. cd to project folder
echo   4. Run:  python main.py
echo.
echo   Other commands:
echo   python main.py --train-only      train only, no server
echo   python main.py --resnet-only     ResNet only  (~22 min)
echo   python main.py --yolo-only       YOLO only    (~90 min)
echo   python main.py --force-manual    use data/manual/ images
echo   python main.py --show-graphs     plot training graphs
echo   python main.py --serve           serve without retraining
echo ============================================================
pause
