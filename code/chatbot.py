import subprocess
from PIL import Image
import io
import numpy as np
import os

# Function to load specific responses from chitchat.txt
def load_specific_responses(file_path):
    specific_responses = {}
    with open(file_path, 'r') as file:
        for line in file:
            query, response = line.strip().split('|')
            specific_responses[query.lower()] = response
    return specific_responses

# Load specific responses from chitchat.txt
script_directory = os.path.dirname(os.path.realpath(__file__))
chitchat_path = os.path.join(script_directory, "chitchat.txt")
specific_responses = load_specific_responses(chitchat_path)

# Open conversation.txt which stores conversation (file is wiped before next run, replace "w" with "a" to append)
conversation_file = open("conversation.txt", "w")

# Main chatbot loop
while True:
    user_input = input("You: ").lower()
    # Keras model
    if user_input.lower() == "keras":
        print(f"You: {user_input}")
        conversation_file.write(f"You: {user_input}\n")
        # Ask the user to enter directory
        image_path = input("Chatbot: Please enter the path to your directory containing the train and test data: ")
        try:
            result = subprocess.run(['python', 'keras-classification-script.py', image_path], capture_output=True, text=True)
            conversation_file.write(f"Chatbot: {result.stdout}")
            print("Chatbot:")
            print(result.stdout)

        except FileNotFoundError:
            print("Chatbot: Error executing program. File not found.")
        except Exception as e:
            print(f"Chatbot: Error processing image - {str(e)}")
    # PyTorch model
    elif user_input.lower() == "pytorch":
        print(f"You: {user_input}")
        conversation_file.write(f"You: {user_input}\n")

        # Ask the user to enter directory
        image_path = input("Chatbot: Please enter the path to your directory containing the train and test data")
        try:
            result = subprocess.run(['python', 'pytorch-classification-script.py', image_path], capture_output=True, text=True)
            conversation_file.write(f"Chatbot: {result.stdout}")
            print("Chatbot:")
            print(result.stdout)

        except FileNotFoundError:
            print("Chatbot: Error executing program. File not found.")
        except Exception as e:
            print(f"Chatbot: Error processing image - {str(e)}")

    elif user_input.lower() in specific_responses:
        conversation_file.write(f"You: {user_input}")
        print(f"Chatbot: {specific_responses[user_input]}")
        # Write conversation to conversation.txt
        conversation_file.write(f"You: {user_input}\n")
        conversation_file.write(f"Chatbot: {specific_responses[user_input]}\n")
        # Quit
        if user_input in ["q", "quit"]:
            break
    else:
        print("Chatbot: I do not know that information.")

# Close conversation file
conversation_file.close()
