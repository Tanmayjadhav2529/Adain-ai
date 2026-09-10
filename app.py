import io
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Import AdaIN modules from utils
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "supersecretkey")
app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "static" / "uploads")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["TEMPLATES_AUTO_RELOAD"] = True
Bootstrap(app)

# Ensure upload directory exists
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg_path = BASE_DIR / "vgg_normalised.pth"
decoder_path = BASE_DIR / "experiment" / "final_exp" / "decoder_final.pth"

encoder = VGGEncoder(str(vgg_path)).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(str(decoder_path), map_location=device))

encoder.eval()
decoder.eval()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    style_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(
            content_feats, style_feats
        )

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == "POST":
        # --- Handle content image ---
        content_file = request.files.get("content")
        if (
            content_file
            and content_file.filename
            and allowed_file(content_file.filename)
        ):
            content_filename = secure_filename(content_file.filename)
            content_file.save(
                os.path.join(app.config["UPLOAD_FOLDER"], content_filename)
            )
        else:
            content_filename = request.form.get("content_path", "").strip()

        # --- Handle style image ---
        style_file = request.files.get("style")
        if (
            style_file
            and style_file.filename
            and allowed_file(style_file.filename)
        ):
            style_filename = secure_filename(style_file.filename)
            style_file.save(
                os.path.join(app.config["UPLOAD_FOLDER"], style_filename)
            )
        else:
            style_filename = request.form.get("style_path", "").strip()

        # --- Validate ---
        if not content_filename:
            error = "Please upload a content image."
        elif not style_filename:
            error = "Please upload a style image."
        else:
            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"], content_filename
            )
            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"], style_filename
            )

            try:
                alpha = float(request.form.get("alpha", 1.0))
                alpha = max(0.0, min(1.0, alpha))  # clamp to [0, 1]

                content_image = Image.open(content_path).convert("RGB")
                style_image = Image.open(style_path).convert("RGB")

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device,
                )

                result_filename = "stylized_" + content_filename
                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], result_filename
                )
                save_image(stylized_image, result_path)

                result_image = result_filename
            except Exception as e:
                import traceback

                error = (
                    f"Style transfer failed: {str(e)}\n{traceback.format_exc()}"
                )

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(str(BASE_DIR / "examples"), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)







