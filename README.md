# StyleForge AI — Arbitrary Neural Style Transfer (AdaIN)

StyleForge AI is a web application for real-time **Arbitrary Neural Style Transfer** powered by PyTorch, Flask, and the **Adaptive Instance Normalization (AdaIN)** algorithm. It allows users to upload any content image and any style reference image, control the style intensity with a dynamic strength slider, and download high-quality stylized results instantly.

---

## About StyleForge AI

Traditional Neural Style Transfer (NST) approaches rely on slow, iterative optimization processes that require minutes to hours per image or demand training a dedicated neural network for every single style.

**StyleForge AI** solves this limitation using **Adaptive Instance Normalization (AdaIN)**:
1. **VGG-19 Encoder**: Extracts deep feature representations from both the content image and the style image.
2. **AdaIN Layer**: Dynamically aligns the mean and standard deviation of content feature activations to match those of the style features in real-time.
3. **Decoupled Decoder**: Inverts the normalized features back into a high-resolution stylized output image.

This architecture enables **instant, arbitrary style transfer** for any un-seen style image without retraining or fine-tuning the model!

---

## Features

- **Arbitrary Style Transfer**: Apply any artistic style (painting, sketch, abstract, etc.) to any target photo without re-training the model.
- **Adjustable Style Strength ($\alpha$)**: Interactively control style blend intensity from `0.0` (pure content) to `1.0` (full style transfer).
- **Modern & Responsive UI**: Futuristic cyberpunk-themed web interface built using Bootstrap 5, Jinja2 templates, and custom CSS.
- **Render & Cloud Ready**: Formatted with `Procfile` and `gunicorn` for deployment on cloud platforms like Render.
- **Cross-Platform Compatibility**: Fully compatible with Windows, Linux, and macOS environments using project-relative `pathlib` paths.

---

## Tech Stack

- **Deep Learning**: PyTorch (`torch`, `torchvision`)
- **Neural Architecture**: VGG19 Encoder + Custom AdaIN Decoder
- **Backend Framework**: Flask, Werkzeug, Flask-WTF, WTForms, Flask-Bootstrap
- **Production Server**: Gunicorn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5

---

## Directory Structure

```text
Adain-ai/
├── Procfile                # Gunicorn deployment configuration for Render
├── README.md               # Project documentation
├── adain_algo.png          # AdaIN algorithm diagram
├── app.py                  # Main Flask application entry point
├── code.ipynb              # Jupyter notebook for experimentation
├── content_data/           # Sample content images dataset
├── examples/               # Example outputs showcase
├── experiment/             # Pre-trained decoder checkpoints
│   └── final_exp/
│       └── decoder_final.pth
├── requirements.txt        # Python dependencies
├── static/                 # CSS, JavaScript, and user upload directory
│   └── uploads/
├── style_data/             # Sample style reference images
├── templates/              # Jinja2 HTML templates (index.html)
├── train.py                # Model training script
├── utils/                  # PyTorch model architectures & AdaIN helper functions
│   ├── models.py
│   └── utils.py
└── vgg_normalised.pth      # Pre-trained VGG19 encoder weights
```

---

## Getting Started

### Prerequisites

- Python **3.9+** or **3.10+** (Python 3.10 / 3.11 recommended)
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/Tanmayjadhav2529/Adain-ai.git
cd Adain-ai
```

### 2. Create and Activate a Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Locally

To launch the web server on your local machine:

```bash
python app.py
```

Open your browser and navigate to:
```text
http://localhost:5000
```

### Usage:
1. Select a **Content Image** (the photo you want to transform).
2. Select a **Style Image** (the artwork whose style you want to copy).
3. Adjust the **Style Strength Slider** ($\alpha$) between `0.0` and `1.0`.
4. Click **Transfer Style** and download the stylized result.

---

## Training a Custom Decoder

If you want to train the decoder on custom datasets:

```bash
python train.py --content_dir content_data --style_dir style_data --epochs 10
```

Arguments:
- `--content_dir`: Path to folder containing content images.
- `--style_dir`: Path to folder containing style images.
- `--vgg`: Path to pre-trained VGG encoder (`vgg_normalised.pth`).
- `--experiment`: Output directory name under `experiment/`.

---

## Deploying to Render

To deploy this project to [Render](https://render.com/):

1. Connect your GitHub repository **`Tanmayjadhav2529/Adain-ai`**.
2. Create a new **Web Service**.
3. Configure the service settings:
   - **Root Directory**: `[Leave Empty]`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`

---

## References & Acknowledgments

- **Paper**: Xun Huang, Serge Belongie. *"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"* (ICCV 2017).
- **VGG Weights**: Normalized VGG-19 network pre-trained on ImageNet.