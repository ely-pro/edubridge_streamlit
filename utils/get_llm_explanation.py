import os
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")
MISTRALAI_MODEL = os.getenv("MISTRALAI_MODEL")


model = MISTRALAI_MODEL
client = Mistral(api_key=MISTRALAI_API_KEY)


def get_llm_explanation(shap_values_df, top_features, dropout_probability, risk_status):
    explanation_template = """
        You are an AI assistant helping explain the dropout risk prediction for a student based on their personal and academic data in the EduBridge system.
        
        The following are the student's features and their respective SHAP values (indicating their importance and direction):
        {shap_values}
        
        The dropout probability predicted by the model is {dropout_probability}%.
        The risk status is {risk_status}.
        
        Your task is to explain in plain language why this dropout risk is high or low based on the most impactful features. The explanation should focus on the following:
        - Which features are most important in predicting dropout risk?
        - How does each important feature affect the dropout prediction? Does it increase or decrease the likelihood of dropout?
        - Why is the student's risk considered high or low?

        Additionally, provide recommendations to help the student improve their chances of staying enrolled, based on the predicted risk.
        
        Please structure your explanation in a clear and concise way for an educator or school administrator to understand.
    """

    # Prepare the explanation with actual data
    explanation_request = explanation_template.format(
        shap_values=shap_values_df.to_string(),
        dropout_probability=dropout_probability,
        risk_status=risk_status
    )

    # Use the Mistral API to generate an explanation
    try:
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": """
                    EduBridge is a transformative AI-driven platform designed to tackle one of Rwanda’s biggest educational challenges: student dropout prevention. 
                    By leveraging advanced data analytics and machine learning, EduBridge predicts and prevents student dropouts, providing early intervention to keep students on track and engaged in their education.
                    
                    Rwanda has made remarkable strides in education, with achievements like Universal Primary Education and net enrollment rates consistently above 97.5%. 
                    Despite these successes, the country still faces challenges with high dropout rates, particularly at critical transition points, such as from primary to secondary school. 
                    This is where EduBridge comes in — helping educators and policymakers identify students at risk of dropping out before it's too late.
                    
                    You are an assistant within the EduBridge system, helping explain dropout risk predictions in an educational context.
                """},
                {"role": "user", "content": explanation_request}
            ]
        )

        # Extract the assistant's response from the model's reply
        assistant_response = chat_response.choices[0].message.content
        return assistant_response

    except Exception as e:
        return f"Error generating response: {e}"
