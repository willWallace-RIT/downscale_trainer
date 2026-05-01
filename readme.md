# Contour-Guided Reconstruction

## Install
pip install -r requirements.txt

## Train
python train.py

## Test
pytest

## Structure
.
├── dataset.py
├── model.py
├── loss.py
├── train.py
├── config.yaml
├── utils/
│   └── config.py
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
