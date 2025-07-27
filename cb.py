from transformers import pipeline

# Use the HuggingFace transformer pipeline for GPT-2 or similar models
generator = pipeline('text-generation', model='gpt2')

def generative_chatbot(user_input):
    prompt = f"User: {user_input}\nChatbot:"
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text'].replace(prompt, "").strip()

# Example usage
print(generative_chatbot("Your replies don't make sense you idiot!"))
