import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox

# ================= LOAD DATA =================
try:
    data = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: data.csv not found!")
    exit()

# ================= TRAIN MODEL =================
X = data[['Temperature', 'Humidity', 'WindSpeed']]
y = data['Energy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42   # ✅ added random_state
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ================= GLOBAL VARIABLES =================
user_prediction = None
user_actual = None
prediction_history = []

# ================= FUNCTIONS =================
def predict_energy():
    global user_prediction, user_actual, prediction_history

    try:
        temp = float(entry_temp.get())
        hum = float(entry_hum.get())
        wind = float(entry_wind.get())

        # 🔴 VALIDATION
        if hum < 0 or hum > 100:
            messagebox.showerror("Invalid Input", "Humidity must be between 0 and 100!")
            entry_hum.delete(0, tk.END)
            return

        if wind < 0:
            messagebox.showerror("Invalid Input", "Wind Speed cannot be negative!")
            entry_wind.delete(0, tk.END)
            return

        # ✅ FIXED: Use DataFrame instead of numpy array
        input_data = pd.DataFrame(
            [[temp, hum, wind]],
            columns=['Temperature', 'Humidity', 'WindSpeed']
        )

        prediction = float(model.predict(input_data)[0])

        user_prediction = prediction

        # ⭐ Store last 3 predictions
        prediction_history.append(round(prediction, 2))
        if len(prediction_history) > 3:
            prediction_history.pop(0)

        # Find closest actual value
        y_array = np.array(y_test, dtype=float)
        differences = np.abs(y_array - prediction)
        closest_index = np.argmin(differences)
        user_actual = float(y_array[closest_index])

        # Show result + history
        history_text = "\n".join(
            [f"{i+1}. {val} units" for i, val in enumerate(prediction_history)]
        )

        result_label.config(
            text=f"🔋 Your Prediction: {prediction:.2f} units\n\n📜 History:\n{history_text}",
            fg="purple"
        )

        # SMART EXPLANATION
        explanation = ""

        if temp > 35:
            explanation += "High temperature increases energy usage.\n\n"
        if hum > 70:
            explanation += "High humidity increases energy consumption.\n\n"
        if wind > 30:
            explanation += "High wind speed may affect energy efficiency.\n\n"

        if prediction < 200:
            explanation += "Overall energy consumption is low.\n\n"
        elif prediction < 500:
            explanation += "Energy consumption is moderate.\n\n"
        else:
            explanation += "Energy consumption is high.\n\n"

        messagebox.showinfo("Analysis", explanation)

    except ValueError:
        messagebox.showerror("Input Error", "Enter valid numeric values!")


def show_graph():
    global user_prediction, user_actual

    if user_prediction is None or user_actual is None:
        messagebox.showerror("Error", "Please click Predict Energy first!")
        return

    sample_X = X_test[:2]
    sample_actual = y_test[:2].values.astype(float)
    sample_predicted = model.predict(sample_X).astype(float)

    actual_values = list(sample_actual) + [user_actual]
    predicted_values = list(sample_predicted) + [user_prediction]

    plt.figure()
    plt.plot(actual_values, predicted_values, marker='o', linestyle='-')

    plt.text(actual_values[0], predicted_values[0], " P1")
    plt.text(actual_values[1], predicted_values[1], " P2")
    plt.text(user_actual, user_prediction, " You", color='purple')

    min_val = min(actual_values + predicted_values)
    max_val = max(actual_values + predicted_values)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel("Actual Energy")
    plt.ylabel("Predicted Energy")
    plt.title("Energy Prediction (2 Dataset + 1 User Point)")
    plt.grid(True)

    plt.show(block=False)
    messagebox.showinfo("Success", "Energy predicted successfully")


def clear_fields():
    global user_prediction, user_actual

    entry_temp.delete(0, tk.END)
    entry_hum.delete(0, tk.END)
    entry_wind.delete(0, tk.END)

    result_label.config(text="")

    user_prediction = None
    user_actual = None


# ================= GUI =================
root = tk.Tk()
root.title("Energy Consumption Predictor")
root.geometry("450x500")
root.config(bg="#f0f4f7")

tk.Label(root, text="Energy Prediction System",
         font=("Arial", 18, "bold"),
         bg="#f0f4f7",
         fg="#C71585").pack(pady=10)

tk.Label(root, text=f"Model Accuracy: {accuracy:.2f}",
         font=("Arial", 10),
         bg="#f0f4f7", fg="blue").pack(pady=5)

frame = tk.Frame(root, bg="#f0f4f7")
frame.pack(pady=10)

tk.Label(frame, text="Temperature (°C)", bg="#f0f4f7").grid(row=0, column=0, pady=5)
entry_temp = tk.Entry(frame)
entry_temp.grid(row=0, column=1)

tk.Label(frame, text="Humidity (%)", bg="#f0f4f7").grid(row=1, column=0, pady=5)
entry_hum = tk.Entry(frame)
entry_hum.grid(row=1, column=1)

tk.Label(frame, text="Wind Speed (km/h)", bg="#f0f4f7").grid(row=2, column=0, pady=5)
entry_wind = tk.Entry(frame)
entry_wind.grid(row=2, column=1)

tk.Button(root, text="Predict Energy",
          command=predict_energy,
          bg="#4CAF50", fg="white",
          width=20).pack(pady=10)

tk.Button(root, text="Show Graph",
          command=show_graph,
          bg="#2196F3", fg="white",
          width=20).pack(pady=5)

tk.Button(root, text="Clear",
          command=clear_fields,
          bg="#f44336", fg="white",
          width=20).pack(pady=5)

result_label = tk.Label(root, text="",
                        font=("Arial", 14),
                        bg="#f0f4f7",
                        justify="left")
result_label.pack(pady=20)

tk.Label(root, text="Mini Project - Energy Prediction",
         font=("Arial", 8),
         bg="#f0f4f7").pack(side="bottom", pady=10)

root.mainloop()