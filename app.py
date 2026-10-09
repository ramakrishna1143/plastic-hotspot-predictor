
from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import sqlite3
import joblib
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# --------------------------------------------------
# FILE PATHS AND MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plastic_hotspot_model.pkl"
DATABASE = BASE_DIR / "database.db"

try:
    model = joblib.load(MODEL_PATH)
    app.logger.info("Plastic hotspot model loaded successfully.")
except Exception:
    app.logger.exception("Could not load the ML model.")
    raise

# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(str(DATABASE))
    connection.row_factory = sqlite3.Row
    return connection


def create_database():
    connection = get_connection()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                people REAL,
                shops REAL,
                bins REAL,
                waste_level INTEGER,
                collection_frequency INTEGER,
                latitude REAL,
                longitude REAL,
                risk TEXT,
                confidence REAL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT,
                latitude REAL,
                longitude REAL,
                risk TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_read INTEGER DEFAULT 0
            )
        """)

        connection.commit()
    finally:
        connection.close()


create_database()

# --------------------------------------------------
# RISK LABEL
# Verify these class labels match train_model.py.
# --------------------------------------------------

def get_risk_label(prediction):
    label_map = {
        0: "LOW RISK",
        1: "MEDIUM RISK",
        2: "HIGH RISK"
    }

    try:
        label = int(prediction)
        return label_map.get(label, str(prediction).upper())
    except (ValueError, TypeError):
        return str(prediction).strip().upper()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    # Read and validate all required form fields.
    required_fields = [
        "people",
        "shops",
        "bins",
        "waste_level",
        "collection_frequency",
        "latitude",
        "longitude"
    ]

    missing_fields = [
        field for field in required_fields
        if not request.form.get(field, "").strip()
    ]

    if missing_fields:
        return (
            "Missing required fields: "
            + ", ".join(missing_fields)
            + ". Please return to the home page and complete the form.",
            400
        )

    try:
        people = float(request.form["people"])
        shops = float(request.form["shops"])
        bins = float(request.form["bins"])
        waste_level = int(request.form["waste_level"])
        collection_frequency = int(
            request.form["collection_frequency"]
        )
        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])

    except (ValueError, TypeError):
        return (
            "Invalid input. Enter valid numbers in all fields "
            "and select valid options.",
            400
        )

    # Validate input ranges.
    if not all(map(
        lambda value: value >= 0 and value < float("inf"),
        [people, shops, bins]
    )):
        return "People, shops, and bins must be non-negative finite numbers.", 400

    if waste_level not in (1, 2, 3):
        return "Invalid waste level. Select an available option.", 400

    if collection_frequency not in (1, 2, 3):
        return "Invalid collection frequency. Select an available option.", 400

    if not (-90 <= latitude <= 90):
        return "Latitude must be between -90 and 90.", 400

    if not (-180 <= longitude <= 180):
        return "Longitude must be between -180 and 180.", 400

    # Feature order must match the model's training order.
    features = [[
        people,
        shops,
        bins,
        waste_level,
        collection_frequency
    ]]

    try:
        prediction = model.predict(features)[0]
        risk = get_risk_label(prediction)

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))

        connection = get_connection()
        try:
            cursor = connection.cursor()

            cursor.execute("""
                INSERT INTO predictions (
                    people, shops, bins, waste_level,
                    collection_frequency, latitude, longitude,
                    risk, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                people, shops, bins, waste_level,
                collection_frequency, latitude, longitude,
                risk, confidence
            ))

            if risk == "HIGH RISK":
                message = (
                    "High plastic-waste risk detected at "
                    f"latitude {latitude}, longitude {longitude}."
                )

                cursor.execute("""
                    INSERT INTO notifications (
                        message, latitude, longitude, risk, confidence
                    )
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    message, latitude, longitude, risk, confidence
                ))

            connection.commit()
        finally:
            connection.close()

        return render_template(
            "result.html",
            result=risk,
            confidence=confidence,
            latitude=latitude,
            longitude=longitude
        )

    except Exception:
        app.logger.exception("Prediction failed.")
        return (
            "The prediction could not be completed. "
            "Please check the application logs.",
            500
        )


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

@app.route("/history")
def history():
    connection = get_connection()
    try:
        rows = connection.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
        """).fetchall()

        predictions = [dict(row) for row in rows]
    finally:
        connection.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


# --------------------------------------------------
# RISK MAP
# Locations format: [latitude, longitude, risk, confidence]
# --------------------------------------------------

@app.route("/map")
def map_view():
    connection = get_connection()
    try:
        rows = connection.execute("""
            SELECT latitude, longitude, risk, confidence
            FROM predictions
            ORDER BY id DESC
        """).fetchall()

        locations = [
            [
                row["latitude"],
                row["longitude"],
                row["risk"],
                row["confidence"]
            ]
            for row in rows
        ]
    finally:
        connection.close()

    return render_template(
        "map.html",
        locations=locations
    )


# --------------------------------------------------
# NOTIFICATIONS
# --------------------------------------------------

@app.route("/notifications")
def notifications():
    connection = get_connection()
    try:
        rows = connection.execute("""
            SELECT *
            FROM notifications
            ORDER BY id DESC
        """).fetchall()

        notification_list = [dict(row) for row in rows]
    finally:
        connection.close()

    return render_template(
        "notifications.html",
        notifications=notification_list
    )


@app.route(
    "/notifications/read/<int:notification_id>",
    methods=["POST"]
)
def mark_notification_read(notification_id):
    connection = get_connection()
    try:
        connection.execute("""
            UPDATE notifications
            SET is_read = 1
            WHERE id = ?
        """, (notification_id,))
        connection.commit()
    finally:
        connection.close()

    return redirect(url_for("notifications"))


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():
    connection = get_connection()
    try:
        total_predictions = connection.execute("""
            SELECT COUNT(*) AS total
            FROM predictions
        """).fetchone()["total"]

        risk_rows = connection.execute("""
            SELECT UPPER(TRIM(risk)) AS risk, COUNT(*) AS total
            FROM predictions
            GROUP BY UPPER(TRIM(risk))
        """).fetchall()

        risk_counts = {
            "LOW RISK": 0,
            "MEDIUM RISK": 0,
            "HIGH RISK": 0
        }

        for row in risk_rows:
            if row["risk"] in risk_counts:
                risk_counts[row["risk"]] = row["total"]

        average_confidence = connection.execute("""
            SELECT AVG(confidence) AS average
            FROM predictions
        """).fetchone()["average"]

        recent_rows = connection.execute("""
            SELECT *
            FROM notifications
            ORDER BY id DESC
            LIMIT 5
        """).fetchall()

        recent_notifications = [
            dict(row) for row in recent_rows
        ]
    finally:
        connection.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        low_count=risk_counts["LOW RISK"],
        medium_count=risk_counts["MEDIUM RISK"],
        high_count=risk_counts["HIGH RISK"],
        average_confidence=(
            average_confidence
            if average_confidence is not None
            else 0
        ),
        recent_notifications=recent_notifications
    )


# --------------------------------------------------
# RUN APPLICATION
# Render runs this app using: gunicorn app:app
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
