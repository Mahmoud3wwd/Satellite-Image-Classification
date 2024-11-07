from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load your trained model
model = tf.keras.models.load_model('my_model_2.h5')

# Define image size
IMG_SIZE = (224, 224)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! Send me a satellite image, and I will classify it for you.')

# Handle image messages
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Get the image file from the user message
    file = await update.message.photo[-1].get_file()
    file_path = 'temp_image.jpg'
    await file.download_to_drive(file_path)  # Use download_to_drive for the latest version

    # Prepare the image for prediction
    img_array = prepare_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Define class labels
    class_labels = ["Cloudy", "Desert", "Green_Area", "Water"]  # Replace with actual class names
    predicted_label = class_labels[predicted_class[0]]

    # Send prediction result back to the user
    await update.message.reply_text(f'Predicted class: {predicted_label}')

    # Clean up by deleting the temp image
    os.remove(file_path)

# Main function to start the bot
def main():
    # Replace 'YOUR_TELEGRAM_BOT_TOKEN' with the token you got from BotFather
    application = Application.builder().token('7908096665:AAGmhFBSPpmBG-EzxXzlf7d6h33Ws6eXFLw').build()

    # Register command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Start polling for updates
    application.run_polling()

if __name__ == '__main__':
    main()
