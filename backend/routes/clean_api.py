from flask import Blueprint, request, jsonify
import os

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "cleaned"

os.makedirs(UPLOAD_DIR, exist_ok = True)
os.makedirs(OUTPUT_DIR, exist_ok = True)

def save_file(file):
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)
    return path

def save_output(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index = False)
    return path

clean_bp = Blueprint("cleaning", __name__)

@clean_bp.route("/upload", methods = ["GET"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message" : "Error! No file uploaded"})

    file = request.files["file"]
    file_path = save_file(file)

    return jsonify({"message" : "File Uploaded Successfully", "path" : file_path})

@clean_bp.route("/clean")
def clean_data():
    filepath = request.json.get("path")
    # cleaned_data, report = clean_data(filepath)
    # output_path = utils.save_output(cleaned_data)
    report = "done"

    return jsonify({"message" : "Data Cleaned Successfully", "report" : report}) # has to add file : output_path