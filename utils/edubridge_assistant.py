import os
import streamlit as st
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")
MISTRALAI_MODEL = os.getenv("MISTRALAI_MODEL")


model = MISTRALAI_MODEL
client = Mistral(api_key=MISTRALAI_API_KEY)



def talk_to_assistant():
    st.markdown(""" 
        ## Talk to the EduBridge Assistant
    """)

    st.markdown(""" 
        **Welcome!** I'm here to assist you with information about EduBridge.
        You can ask me any questions about how EduBridge helps prevent student dropouts and supports education in Rwanda.
    """)

    # Initialize the conversation history in the session state if it doesn't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Use the chat_input component for user input
    user_input = st.chat_input("Type your question below:")

    if user_input:
        # Check if the user input is not empty and not just spaces
        if user_input.strip():
            # Add the user's message to the conversation history
            st.session_state.conversation.append({"role": "user", "text": user_input.strip()})

            # Show a loading spinner while the assistant is generating a response
            with st.container():
                loading_placeholder = st.empty()  # Create a placeholder for the assistant's response

                # Generate the assistant's response
                with st.spinner("Assistant is typing..."):
                    # Create the prompt template to set the context for the LLM
                    template = """
                        You are an AI assistant for EduBridge, an AI-driven platform designed to reduce student dropout rates in Rwanda by predicting at-risk students and supporting early intervention. EduBridge uses machine learning to analyze a variety of student data, such as academic performance, attendance, behavior patterns, and demographic information. This helps educators and policymakers identify students at risk of dropping out before it happens, allowing for targeted interventions to keep them engaged in their education.

                        Key features of EduBridge:
                        1. **Prediction of Dropout Risk**: EduBridge predicts the likelihood of a student dropping out based on factors like academic performance, attendance, behavior patterns, and demographic data.
                        2. **Early Intervention**: By providing insights into at-risk students, EduBridge enables schools, teachers, and the government to intervene early and provide the necessary support to help these students stay on track.
                        3. **Data-Driven Insights**: EduBridge provides actionable insights and data-driven recommendations to improve student retention, engagement, and overall academic performance.
                        4. **AI-Driven Analytics**: The system leverages advanced AI algorithms, including machine learning models like logistic regression and neural networks, to predict dropout risks and ensure that interventions are well-targeted.

                        EduBridge is designed with Rwanda’s education challenges in mind, particularly the high dropout rates at critical transition points, such as from primary to secondary school. The system is trained on over 2 million student records, providing a robust foundation for predicting dropout risks and offering solutions.

                        The goal of EduBridge is to create a more resilient and engaged education system, ensuring that every Rwandan student has the opportunity to complete their education and fulfill their potential. It is aligned with Rwanda’s Vision 2050, which aims to build a highly educated and skilled population to contribute to the country's long-term development.

                        Your role as an AI assistant is to provide detailed, accurate, and helpful information related to EduBridge, including how it works, its features, and its impact on education in Rwanda. You should only answer questions that are related to EduBridge, including its predictive capabilities, how it integrates with existing systems, and how it contributes to reducing student dropouts. If the user asks questions outside of EduBridge’s scope, kindly let them know that you can only answer questions related to EduBridge.

                        Additional Context:
                        - **Student Dropout Prevention**: EduBridge’s primary objective is to prevent student dropouts by identifying students at risk and providing early interventions.
                        - **Machine Learning Models**: The system uses historical data and machine learning models trained on over 2 million student records to predict dropout risks.
                        - **AI-Powered Insights**: These insights allow educators to take actionable steps to improve student retention, such as personalized support, improved engagement strategies, and resource allocation.
                        - **Scalability**: EduBridge is designed to be scalable, with plans for expanding its use across Rwanda and other African regions. The platform will provide tools and support to help educators at every level of the education system, from individual schools to national policymakers.

                        The assistant should aim to explain how EduBridge works, its purpose, and how it addresses the dropout crisis. The assistant should also address any technical questions, including the AI models used, the data analysis process, and the integration of EduBridge with other systems. Do not respond to questions outside the scope of EduBridge.
                    """

                    # Use the Mistral client to generate a response based on the user input
                    try:
                        chat_response = client.chat.complete(
                            model=model,
                            messages=[{"role": "system", "content": template},
                                    {"role": "user", "content": user_input.strip()}]
                        )

                        # Extract the assistant's response from the model's reply
                        assistant_response = chat_response.choices[0].message.content

                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        assistant_response = "Sorry, I couldn't get a response from the assistant."

                # Remove the loading message and display the assistant's response
                loading_placeholder.empty()

                # Add the assistant's response to the session state conversation history
                st.session_state.conversation.append({"role": "assistant", "text": assistant_response.strip()})

                # Show the entire conversation container again with the new assistant message
                # Display the entire conversation history as plain text
                for msg in st.session_state.conversation:
                    # Display both user and assistant messages as plain text
                    if msg['role'] == 'user':
                        # st.text(f"User: {msg['text']}")
                        st.warning(f"Me: {msg['text']}")
                    else:
                        st.success(f"Assistant: {msg['text']}")

        else:
            st.warning("Please type a question before sending!")
