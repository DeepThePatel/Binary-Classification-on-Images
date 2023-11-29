#!/usr/bin/env python
# coding: utf-8

import subprocess
import os

# Function to load specific responses from chitchat.txt
def load_specific_responses(file_path):
    specific_responses = {}
    with open(file_path, 'r') as file:
        for line in file:
            query, response = line.strip().split('|')
            specific_responses[query.lower()] = response
    return specific_responses

# Determine the absolute path to chitchat.txt
script_directory = os.path.dirname(os.path.realpath(__file__))
chitchat_path = os.path.join(script_directory, "chitchat.txt")

# Load specific responses from chitchat.txt
specific_responses = load_specific_responses(chitchat_path)

# Open conversation.txt which stores conversation (file is wiped before next run, replace "w" with "a" to append)
conversation_file = open("conversation.txt", "w")

# Main chatbot loop
while True:
    user_input = input("You: ").lower()

    if user_input == "execute":
        conversation_file.write(f"You: {user_input}\n")
        try:
            result = subprocess.run(['python', 'keras-classification-script.py'], capture_output=True, text=True)
            print("Program Output:")
            print(result.stdout)
        except FileNotFoundError:
            print("Chatbot: Error executing program. File not found.")
    elif user_input == "execute2":
        conversation_file.write(f"You: {user_input}\n")
        try:
            result = subprocess.run(['python', 'pytorch-classification-script.py'], capture_output=True, text=True)
            print("Program Output:")
            print(result.stdout)
        except FileNotFoundError:
            print("Chatbot: Error executing program. File not found.")
    elif user_input in specific_responses:
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

