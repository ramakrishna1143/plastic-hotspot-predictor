
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import joblib
from datetime import datetime

app = Flask(__name__)

# --------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------------------------

model = joblib.load("plastic_hotspot_model.pkl")

DATABASE = "database.db"


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


# --------------------------------------------------
# CREATE DATABASE TABLES
# --------------------------------------------------

def create_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
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

    cursor.execute("""
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
    connection.close()


create_database()


# --------------------------------------------------
# RISK LABEL HELPER
# --------------------------------------------------

def get_risk_label(prediction):
    """
    Convert model output into a readable risk label.

    Confirm these class labels match your model's training.
    """
    label_map = {
        0: "LOW RISK",
        1: "MEDIUM RISK",
        2: "HIGH RISK"
    }

    try:
        numeric_label = int(prediction)
        return label_map.get(numeric_label, str(prediction).upper())
    except (ValueError, TypeError):
        return str(prediction).strip().upper()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION ROUTE
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
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

        # Validate numeric inputs
        if people < 0 or shops < 0 or bins < 0:
            return "People, shops, and bins cannot be negative.", 400

        if waste_level not in (1, 2, 3):
            return "Invalid waste level.", 400

        if collection_frequency not in (1, 2, 3):
            return "Invalid collection frequency.", 400

        if not -90 <= latitude <= 90:
            return "Invalid latitude.", 400

        if not -180 <= longitude <= 180:
            return "Invalid longitude.", 400

        # IMPORTANT:
        # Input order must match the order used when training the model.
        features = [[
            people,
            shops,
            bins,
            waste_level,
            collection_frequency
        ]]

        prediction = model.predict(features)[0]
        risk = get_risk_label(prediction)

        # Get prediction confidence when supported
        confidence = 0.0

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO predictions (
                people,
                shops,
                bins,
                waste_level,
                collection_frequency,
                latitude,
                longitude,
                risk,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            people,
            shops,
            bins,
            waste_level,
            collection_frequency,
            latitude,
            longitude,
            risk,
            confidence
        ))

        # Create a notification for high-risk predictions
        if risk == "HIGH RISK":
            message = (
                f"High plastic-waste risk detected at "
                f"latitude {latitude}, longitude {longitude}."
            )

            cursor.execute("""
                INSERT INTO notifications (
                    message,
                    latitude,
                    longitude,
                    risk,
                    confidence
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                message,
                latitude,
                longitude,
                risk,
                confidence
            ))

        connection.commit()
        connection.close()

        return render_template(
            "result.html",
            result=risk,
            confidence=confidence,
            latitude=latitude,
            longitude=longitude
        )

    except (ValueError, TypeError, KeyError):
        return (
            "Invalid or missing form values. "
            "Please check all fields and try again.",
            400
        )

    except Exception as error:
        app.logger.exception("Prediction failed")
        return "Prediction failed. Check the VS Code terminal for details.", 500


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

        # Convert rows to dictionaries for template compatibility
        predictions = [dict(row) for row in rows]

    finally:
        connection.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


# --------------------------------------------------
# RISK MAP
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

        # FIX: SQLite Row objects are not JSON serializable.
        # Convert them into normal Python lists.
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
# NOTIFICATIONS PAGE
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

        notification_list = [
            dict(row) for row in rows
        ]

    finally:
        connection.close()

    return render_template(
        "notifications.html",
        notifications=notification_list
    )


# --------------------------------------------------
# MARK NOTIFICATION AS READ
# --------------------------------------------------

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
            risk_name = row["risk"]
            if risk_name in risk_counts:
                risk_counts[risk_name] = row["total"]

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
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
