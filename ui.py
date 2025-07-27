import streamlit as st

def chatbot():
    st.title("Chatbot with History")
    st.markdown("### Chat with me and see the history of our conversation.")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You:", "")

    if user_input:
        st.session_state.history.append(f"You: {user_input}")
        
        bot_response = f"I'm here to help! You said: {user_input}"
        st.session_state.history.append(f"Bot: {bot_response}")

    if st.session_state.history:
        st.write("### Conversation History:")
        for message in st.session_state.history:
            st.write(message)

if __name__ == "__main__":
    chatbot()
