import argparse
from pathlib import Path

import joblib


def main():
    parser = argparse.ArgumentParser(
        description="Classify a banking support message."
    )
    parser.add_argument("message", help="Customer message in quotes")
    args = parser.parse_args()

    message = args.message.strip()

    if not message:
        parser.error("Please provide a non-empty message.")

    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "ticket_classifier.joblib"
    )

    if not model_path.is_file():
        parser.error(
            "Model file missing. Run the training notebook "
            "through the model-saving step first."
        )

    model = joblib.load(model_path)
    prediction = model.predict([message])[0]

    print(f"Message: {message}")
    print(f"Predicted category: {prediction}")


if __name__ == "__main__":
    main()