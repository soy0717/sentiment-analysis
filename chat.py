import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# Initialize the model and tokenizer for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize the model and tokenizer for DialoGPT
model_name = "microsoft/DialoGPT-small"  # Using medium model for better responses
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

MAX_HISTORY_LENGTH = 1000
MIN_RESPONSE_LENGTH = 10  # Minimum response length to avoid short/incomplete responses

def preprocess(text):
    tokens = word_tokenize(text.lower())
    lem = WordNetLemmatizer()
    lemmatized_tokens = [lem.lemmatize(token) for token in tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def sentiment_analysis(text):
    result = sentiment_pipeline(text)
    sentiment = result[0]['label'].lower()
    scores = {'confidence': result[0]['score']}
    return sentiment, scores

def adjust_generation_params(sentiment, confidence):
    """Adjust generation parameters based on sentiment and confidence"""
    if sentiment == "positive":
        return {
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.2,
            'min_length': MIN_RESPONSE_LENGTH
        }
    elif sentiment == "negative":
        return {
            'temperature': 0.6,  # More conservative for negative sentiment
            'top_p': 0.85,
            'top_k': 40,
            'repetition_penalty': 1.3,
            'min_length': MIN_RESPONSE_LENGTH
        }
    else:
        return {
            'temperature': 0.7,
            'top_p': 0.87,
            'top_k': 45,
            'repetition_penalty': 1.2,
            'min_length': MIN_RESPONSE_LENGTH
        }

def generate_response(prompt, conversation_history=None):
    # Create attention mask
    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long)
    
    # Handle conversation history
    if conversation_history is not None:
        bot_input_ids = torch.cat([conversation_history, new_input_ids], dim=-1)
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
    else:
        bot_input_ids = new_input_ids
    
    # Trim history if too long
    if bot_input_ids.shape[-1] > MAX_HISTORY_LENGTH:
        bot_input_ids = bot_input_ids[:, -MAX_HISTORY_LENGTH:]
        attention_mask = attention_mask[:, -MAX_HISTORY_LENGTH:]
    
    # Generate multiple responses and pick the best one
    num_responses = 3
    responses = []
    
    for _ in range(num_responses):
        chat_history_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=MAX_HISTORY_LENGTH,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            repetition_penalty=1.2,
            length_penalty=1.0,
            min_length=MIN_RESPONSE_LENGTH
        )
        
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        responses.append((response, chat_history_ids))
    
    # Filter out empty or very short responses
    valid_responses = [(r, h) for r, h in responses if len(r.strip()) >= MIN_RESPONSE_LENGTH]
    
    if not valid_responses:
        return "I'm processing that. Could you rephrase?", conversation_history
    
    # Select the best response (longest valid response)
    best_response, best_history = max(valid_responses, key=lambda x: len(x[0].strip()))
    
    return best_response.strip(), best_history

class Chatbot:
    def __init__(self):
        self.conversation_history = None
        self.consecutive_short_responses = 0
        self.last_response = None

    def get_response(self, user_input):
        cleaned_input = preprocess(user_input)
        sentiment, scores = sentiment_analysis(cleaned_input)

        # Dynamic sentiment-based prompts
        sentiment_prompts = {
            "negative": [
                "I understand you're feeling down. ",
                "I hear that you're having a difficult time. ",
                "Let me support you. ",
                "I'm here to listen. "
            ],
            "positive": [
                "I'm glad you're feeling good! ",
                "That's wonderful! ",
                "I'm happy to hear that! ",
                "That's great! "
            ],
            "neutral": [
                "I understand. ",
                "I hear you. ",
                "Let's discuss that. ",
                "Tell me more. "
            ]
        }

        # Select a random prompt based on sentiment
        prompt = np.random.choice(sentiment_prompts[sentiment]) + cleaned_input
        
        # Generate response with adjusted parameters
        response, new_history = generate_response(prompt, self.conversation_history)
        
        # Check response quality
        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            self.consecutive_short_responses += 1
            if self.consecutive_short_responses >= 2:
                # Reset conversation if getting too many short responses
                self.conversation_history = None
                self.consecutive_short_responses = 0
                response = "Let me start fresh. " + response
        else:
            self.consecutive_short_responses = 0
        
        # Update conversation history
        self.conversation_history = new_history
        self.last_response = response

        return {
            "sentiment": sentiment,
            "confidence": scores['confidence'],
            "response": response
        }

def main():
    print("Initializing chatbot...")
    chatbot = Chatbot()
    print("Chatbot initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        if user_input.lower() == 'reset':
            chatbot.conversation_history = None
            print("Conversation reset.")
            continue
            
        try:
            result = chatbot.get_response(user_input)
            print(f"\nBot: {result['response']}")
            # Uncomment for debugging
            # print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            chatbot.conversation_history = None
            continue

if __name__ == "__main__":
    main()