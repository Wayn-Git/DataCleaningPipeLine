from clean_api import save_file, save_output
from flask import Blueprint, request, jsonify

augment_bp = Blueprint("augmentation", __name__)

@augment_bp.route("/augment", methods = ["GET"])
def augment_data():
    filepath = request.json.get("path")
    # augmented_df, report = augment()
    # output_path = save_output(augmented_df, "augmented_data.csv")
    report = "done"
    return jsonify({"message" : "Data Augmented Successfully", "report" : report})# has to add file : output_path