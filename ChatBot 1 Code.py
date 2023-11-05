import random

rules = {
    "hello": "Hello! How can I help you?",
    "how are you": "I'm just a chatbot, but I'm here to assist you. What can I do for you?",
    "check my account balance": "Of course! I'd be happy to assist you. To check your account balance, please provide your account number to receive a verification code",
    "new user": "Certainly, setting up a recovery number is a great idea for added security. To proceed, please provide your full name and your number for verification. Once verified, we can assist you in setting up a recovery number.",
    "dido godana": "Thank you for providing your information, [Dido Godana]. I see you're a new user. To set up a recovery number, please enter a 6 to 10-digit number that you'll remember.",
    "0796128364": "Great! Your recovery number has been set to 0796128364. This will help you in case you forget your account number in the future. Please make sure to keep it secure. Is there anything else I can assist you with today?",
    "premium payment last month": "Dear customer, we are in receipt of your premium payment. Our system is under maintenace process till end month. You will be get a notification via email or sms. Is there anything else I can assist you with today?",
    "that's all": "You're very welcome, [Dido Godana]! If you have any more questions or need assistance in the future, don't hesitate to reach out. We value your feedback. On a scale of 1 to 5, how would you rate your experience today?",
    "5": "That's great to hear! Thank you for the positive feedback. We're here to provide you with the best service. If you have any more inquiries in the future, please feel free to contact us. Have a wonderful day",
    "goodbye": "Goodbye! Have a great day!",
    "default_response": "I'm not sure how to respond to that. Can you please rephrase your question?"
}

def get_response(user_input):
    user_input = user_input.lower()
    for rule, response in rules.items():
        if rule in user_input:
            return response
    return "I'm sorry, I don't understand.Can you please rephrase that?"

# Main loop to interact with the chatbot
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break
    response = get_response(user_input)
    print("Chatbot:", response)

