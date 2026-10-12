import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train_model():
    try:
        # Update user that training is starting
        messagebox.showinfo("Training Started", "Model training has started. This may take a few seconds...")

        # Data directories
        train_path = "Dataset/train"
        test_path = "Dataset/test"

        # Load images
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_set = train_datagen.flow_from_directory(
            train_path,
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical'
        )

        test_set = test_datagen.flow_from_directory(
            test_path,
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical'
        )

        # Build model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(train_set.num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        history = model.fit(
            train_set,
            validation_data=test_set,
            epochs=10,
            verbose=0  # Hide terminal output
        )

        # Save model
        model.save("sign_model.h5")

        # Training complete pop-up with final accuracy
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        messagebox.showinfo("Training Completed", 
                            f"Model trained and saved successfully!\n\n"
                            f"Train Accuracy: {final_train_acc:.2f}\n"
                            f"Validation Accuracy: {final_val_acc:.2f}")
    except Exception as e:
        messagebox.showerror("Training Failed", f"An error occurred:\n{str(e)}")

# GUI Setup
root = tk.Tk()
root.title("Sign Language Model Trainer")
root.geometry("350x200")

label = tk.Label(root, text="Sign Language Recognition", font=("Helvetica", 16, "bold"))
label.pack(pady=20)

train_button = tk.Button(root, text="Train Model", font=("Helvetica", 14), bg="#4CAF50", fg="white", command=train_model)
train_button.pack(pady=20)

root.mainloop()