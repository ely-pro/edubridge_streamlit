import os
import streamlit as st
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
from dotenv import load_dotenv
from utils.edubridge_assistant import talk_to_assistant
from utils.get_llm_explanation import get_llm_explanation
load_dotenv()

loaded_model = joblib.load("./data/model/lg-model-v2/logistic_regression_dropout_model.pkl")
loaded_preprocessor = joblib.load("./data/model/lg-model-v2/preprocessor.pkl")

masker = joblib.load("./data/model/lg-model-v2/explainer_masker.pkl")
explainer = shap.Explainer(loaded_model, masker)



st.set_page_config(page_title="EduBridge | Dropout Prediction", layout="wide", page_icon="./media/icon/android-chrome-192x192.png")
st.markdown("""
    ## EduBridge | Student Dropout Prediction System
""")

with st.sidebar:
    st.image('./media/img/edubridge/logo/edubridge.png', use_container_width=True)
    st.header("EduBridge")
    st.markdown("""
        EduBridge is a student dropout prediction tool designed to assist Rwandan educators and policymakers in addressing dropout risks.
        The tool provides insights to guide targeted interventions, leveraging a model trained on over 2 million student records.
    """)

tabs = st.tabs(["Overview", "EduBridge In Action", "Assistant", "Developers", "Contact Us"])
  
def model_training_tab():
    st.subheader("ML Model Training")
    
    st.write("""
        In this section, we discuss the process of model training to predict student dropout risks.
        Our approach involves exploring the data, analyzing feature distributions, and applying statistical methods
        to understand the relationships among features. This process is crucial for ensuring that the model accurately
        learns patterns relevant to predicting dropout risks, while generalizing well to new data.
    """)
    
    st.subheader("Feature Correlation Heatmap")
    st.write("""
        The **Feature Correlation Heatmap** shows the correlations between different features in the dataset.
        It is a visual tool that helps us understand how different features relate to one another. 
        Each square in the heatmap represents the correlation coefficient between two features. 
        Correlation values range from -1 to 1:
        
        - **Positive Correlation (0 to 1)**: Features move in the same direction. 
          For example, if "Attendance Rate" and "Average Grade" have a strong positive correlation, 
          it means that as attendance rate increases, the student grades also increase (or vice versa).
          
        - **Negative Correlation (-1 to 0)**: Features move in opposite directions. 
          For example, "Attendance Rate" and "Repetitions in Class" might have a negative correlation: 
          as the attendance rate increases, repetitions in class may decrease.

        - **No Correlation (0)**: There is no predictable relationship between the features.
        
        Identifying strong correlations helps us understand how features interact and informs feature selection 
        for the model. Features that are highly correlated might be redundant, so we often select only one to include 
        in the model to avoid overfitting.
    """)
    st.image('./media/training-graph/v2/correlation-heatmap.png', use_container_width=True)
    
    # Display the boxplot for feature distributions
    st.subheader("Feature Distribution - Boxplot")
    st.write("""
        The **Boxplot** provides a visualization of the spread and distribution of each feature in the dataset.
        It shows the **median**, **quartiles**, and **outliers** of each feature. Here's how to interpret the boxplot:
        
        - **The Box**: The box represents the **interquartile range (IQR)**, which is the middle 50% of the data. 
          The line in the middle of the box is the **median** (the 50th percentile), which indicates the center of the data distribution.
        
        - **Whiskers**: The lines extending from the box (the whiskers) represent the range of the data, typically 
          from the **25th percentile** (lower whisker) to the **75th percentile** (upper whisker), excluding outliers.
        
        - **Outliers**: Points that fall outside the whiskers (typically 1.5 times the IQR) are marked as individual dots.
          These are **outliers**, and they may indicate unusual or extreme values that could be worth investigating.
          
        - **Skewness**: If the box is asymmetric or if the whiskers are of different lengths, this indicates **skewness** in the data. 
          A long upper whisker suggests the data is positively skewed, while a long lower whisker indicates negative skewness.
        
        Analyzing the boxplot helps us understand the **spread** of the data and detect any **outliers** that may require further attention or transformation.
    """)
    st.image('./media/training-graph/v2/box-plot-quartiles.png', use_container_width=True)
    
    # Display the categorical features analysis
    st.subheader("Categorical Feature Distribution")
    st.write("""
        The **Categorical Feature Distribution** plot visualizes how categorical variables are distributed across different 
        categories. For example, this plot might show the number of students in each **school category**, such as **Primary** or **Secondary**. 
        Each category is represented as a bar, and the height of the bar represents the number of observations (e.g., the number of students 
        in each category).
        
        - **Key Insights**: Categorical plots help us understand the **distribution** of different categories and reveal any **imbalances** 
          or **biases** in the data. For instance, if the number of students in "Primary" is much higher than in "Secondary", 
          this could affect how the model handles the prediction task for these two categories.
        
        - **Imbalance**: If some categories are significantly more frequent than others, we might need to consider **balancing techniques** 
          (e.g., oversampling or undersampling) to prevent the model from being biased towards the more frequent categories.
    """)
    st.image('./media/training-graph/v2/categorical-feature-distribution.png', use_container_width=True)
    
    # Display the histogram of numerical feature distributions
    st.subheader("Numerical Feature Distribution")
    st.write("""
        The **Numerical Feature Distribution** histogram shows the frequency distribution of numerical features. 
        Each bar represents a **range of values** for a given feature, and the height of the bar shows how many data points fall 
        within that range. Here’s what we can infer from the histogram:
        
        - **Skewness**: If the histogram is **skewed** to the left or right, it suggests that the data is not symmetrically distributed.
          For example, a **right skew** might indicate that most of the data points are concentrated in lower values, 
          but there are some extreme higher values (a few students with very high grades or very high absenteeism).
        
        - **Range**: The histogram also gives us a sense of the **range** and **distribution** of the data for each numerical feature.
        
        - **Outliers**: Extreme values might appear as isolated bars far away from the rest of the distribution. These outliers could influence the model’s 
          performance, so it’s essential to understand their impact and whether they should be removed or transformed.
          
        - **Normal Distribution**: If the histogram shows a roughly **bell-shaped curve**, it suggests that the feature follows a **normal distribution**. 
          Many machine learning models perform better when features are normally distributed.
    """)
    st.image('./media/training-graph/v2/numerical-feature-distribution.png', use_container_width=True)
    
    st.write("""
        By leveraging these analyses, we selected important features and set up the training process to maximize
        the model's predictive performance while minimizing overfitting.
    """)

def developers_tab():
    col1, col2 = st.columns([1, 3])    
    with col1:
        st.image('./media/img/company/atas.svg', width=160)  
    
    with col2:
        st.write("""
            **ATAS** (Alliance for Transformative AI Systems) is a forward-thinking institution dedicated to advancing the field of artificial intelligence (AI). Our mission is to create AI-driven systems that address pressing global challenges while promoting ethical practices and inclusivity.

            We strive to drive positive societal change through AI-powered innovations across various sectors, including healthcare, education, agriculture, and environmental conservation.

            Our focus is to foster collaboration, ethical AI development, and sustainable impact on a global scale. ATAS works with researchers, technologists, policymakers, and educators to create transformative AI systems that benefit all of humanity.
        """)

    st.markdown("""
        ### 🔍 What we do:
    """)
    st.write("""
        - Conduct cutting-edge research in AI, focusing on machine learning, deep learning, NLP, and computer vision.
        - Develop scalable and impactful AI solutions for sectors like healthcare, education, and the environment.
        - Foster global collaboration to tackle critical global challenges through AI.
        - Promote ethical AI development, ensuring fairness, transparency, and inclusivity in all our initiatives.
        
        # **Contact**: [Website](https://atas.org) | [Email](mailto:contact@atas.org) | [GitHub](https://github.com/atas) | [Instagram](https://instagram.com/atas_ai)
    """)
 
def send_email(subject, message, sender_email):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")
    login_email = os.getenv("LOGIN_EMAIL")
    login_password = os.getenv("LOGIN_PASSWORD")

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = login_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(login_email, login_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"An error occured: {e}")
        return False
    
def contact_us_tab():
    st.subheader("Contact Us")
    st.write("Get in touch with us via the contact details provided or fill out the form to send us a message directly.")

    contact_col, form_col = st.columns(2)

    with contact_col:
        st.write("### Our Contact Information")

        with st.expander("Address"):
            st.write("29 Street, Kigali, Rwanda")
            st.write("Our main office is located in the heart of Kigali, providing easy access to educational resources and support.")

        with st.expander("Phone"):
            st.write("+250 788880206")
            st.write("Feel free to reach us by phone during office hours. We're here to help you with any question you may have.")

        with st.expander("Email"):
            st.write("contact@edubridge.com")
            st.caption("For inquiries, partnerships, or support, send us an email, and our team will respond promptly.")

        with st.expander("WhatsApp"):
            st.write("Chat with us on WhatsApp: [WhatsApp](https://wa.me/250788880206)")
            st.caption("Connect with us on WhatsApp for quick responses to your questions and assistance with our services.")

        with st.expander("Instagram"):
            st.write("Follow us on Instagram: [@edubridge](https://www.instagram.com/edubridge/)")
            st.caption("Stay updated on our latest news, events, and educational resources by following our Instagram page.")

        with st.expander("X"):
            st.write("Follow us on X: [@edubridge](https://www.instagram.com/edubridge/)")
            st.caption("Stay updated on our latest news, events, and educational resources by following our X page.")
    
    with form_col:
        st.write("### Send us a message")
        is_sending = st.session_state.get("is_sending", False)

        with st.form("contact_form"):
            name = st.text_input("Your Name", disabled=is_sending)
            form_col1, form_col2 = st.columns(2)

            with form_col1:
                sender_email = st.text_input("Your Email", disabled=is_sending)

            with form_col2:
                subject = st.text_input("Subject", disabled=is_sending)

            message = st.text_area("Message", disabled=is_sending)
            submitted = st.form_submit_button("Send Message", disabled=is_sending)

            if submitted:
                if name and sender_email and subject and message:
                    st.session_state["is_sending"] = True
                    with st.spinner("Sending message..."):
                        time.sleep(2)
                        success = send_email(
                            subject,
                            f"Name: {name}\nEmail: {sender_email}\n\n{message}",
                            sender_email
                        )

                        if success:
                            st.success("Your message has been sent successfully!")
                        else:
                            st.error("Failed to send your message.")

                    st.session_state["is_sending"] = False
                else:
                    st.warning("Please fill in all fields.")

with tabs[0]:
    st.markdown("""
        ### EduBridge – Empowering Rwanda’s Educational Future
    """)
    st.video("./media/video/EduBridge.mp4")
    st.write("""
        EduBridge is a transformative AI-driven platform designed to tackle one of Rwanda’s biggest educational challenges: **student dropout prevention**. By leveraging advanced data analytics and machine learning, EduBridge predicts and prevents student dropouts, providing early intervention to keep students on track and engaged in their education.

        Rwanda has made remarkable strides in education, with achievements like **Universal Primary Education** and net enrollment rates consistently above 97.5%. Despite these successes, the country still faces challenges with high dropout rates, particularly at critical transition points, such as from primary to secondary school. This is where **EduBridge** comes in — helping educators and policymakers identify students at risk of dropping out before it's too late.
    """)
    st.markdown("""
        ### The Problem: The Persistent Challenge of Student Dropouts
    """)
    st.image("./media/img/edubridge/image0_0 (8).jpg")
    st.write("""
        - **High Dropout Rates**: Dropouts remain a serious concern in Rwanda’s education system, especially among older students who face academic, economic, and social challenges.
        - **Delayed Interventions**: Current systems detect dropouts only after they occur, which limits the effectiveness of interventions and makes it difficult to address the root causes early enough.
    """)

    @st.cache_data
    def get_dropout_data():
        DATA_FILENAME = Path(__file__).parent / './data/dataset/dropout_data.csv'
        dropout_df = pd.read_csv(DATA_FILENAME)

        # Ensure correct data types
        dropout_df['Year'] = pd.to_numeric(dropout_df['Year'])
        dropout_df['Dropout Rate (%)'] = pd.to_numeric(dropout_df['Dropout Rate (%)'], errors='coerce')
        return dropout_df

    dropout_df = get_dropout_data()


    # Year range slider
    min_year = dropout_df['Year'].min()
    max_year = dropout_df['Year'].max()

    from_year, to_year = st.slider(
        'Select the year range:',
        min_value=min_year,
        max_value=max_year,
        value=[min_year, max_year]
    )

    # Education level filter
    education_levels = dropout_df['Education Level'].unique()
    selected_levels = st.multiselect(
        'Select Education Level(s):',
        options=education_levels,
        default=education_levels
    )

    # Filter the data
    filtered_dropout_df = dropout_df[
        (dropout_df['Year'] >= from_year) &
        (dropout_df['Year'] <= to_year) &
        (dropout_df['Education Level'].isin(selected_levels))
    ]

    # Chart: Dropout Rate Over Time
    st.markdown("""### Dropout Rate Over Time""")
    st.line_chart(
        filtered_dropout_df,
        x='Year',
        y='Dropout Rate (%)',
        color='Gender'
    )

    ''
    ''

    # Metrics for the latest year
    st.markdown(f"""### Dropout Rates in {to_year}""")
    cols = st.columns(4)

    for i, (gender, level) in enumerate(
            filtered_dropout_df[filtered_dropout_df['Year'] == to_year].groupby(['Gender', 'Education Level'])):
        col = cols[i % len(cols)]
        with col:
            rate = level['Dropout Rate (%)'].mean()
            st.metric(
                label=f"{gender}",
                value=f"{rate:.2f}" if not math.isnan(rate) else 'N/A'
            )

    
    st.markdown("""
        ### The Solution: EduBridge
    """)
    st.image("./media/img/edubridge/image1_0 (6).jpg")

    st.write("""
        **EduBridge** addresses this problem by using AI and predictive modeling to identify students at risk of dropping out, enabling timely and targeted interventions. This platform is not just a tool for educators, but also for parents and policymakers, helping them to collaborate in the mission to keep students in school and engaged.

        ### Key Features of EduBridge:
        - **Predictive AI Model**: EduBridge uses machine learning algorithms to analyze data from various sources (attendance, grades, behavior, family background) to predict dropout risk. This enables early intervention and proactive support for at-risk students.
        - **Integration with Smart Vision Attendance System (SVAS)**: EduBridge integrates seamlessly with **SVAS**, a cutting-edge attendance tracking system that uses camera-based technology to automatically monitor student attendance, ensuring accurate, real-time data without added workload for teachers.
        - **Multilingual Support**: The platform provides insights and recommendations in multiple languages, making it accessible to a diverse audience, including parents, educators, and policymakers across Rwanda.
    """)

    model_training_tab()

    st.write("""
        
        ### How EduBridge and SVAS Work Together:
        - **AI-Powered Risk Prediction**: EduBridge’s machine learning models (such as Random Forest, Neural Networks, and Logistic Regression) analyze patterns in attendance, academic performance, behavior, and family background to generate risk profiles for each student.
        - **Smart Vision Attendance System (SVAS)**: SVAS enhances dropout prevention by accurately tracking student attendance. Attendance patterns are one of the strongest indicators of potential dropout, and SVAS helps detect irregularities before they become chronic, allowing schools to intervene early.
        - **SHAP for Transparency**: EduBridge uses **Shapley Additive Explanations (SHAP)** to provide stakeholders with transparent, interpretable insights into the key factors driving each dropout prediction, making the intervention process well-informed.

        ### The Impact of EduBridge:
        - **For Schools**: EduBridge helps school administrators monitor dropout risks in real time. With accurate, real-time data from SVAS and predictive models, schools can address academic, attendance, and behavioral issues before they result in dropout.
        - **For Parents**: EduBridge empowers parents with insights into their child's attendance and performance, allowing them to take timely actions to support their child’s education.
        - **For Policymakers**: The platform provides valuable data that can guide policy decisions and resource allocation, contributing to Rwanda’s long-term educational goals.

        ### National and Economic Impact:
        EduBridge contributes to **Rwanda’s Vision 2050** by improving human capital development. By preventing student dropouts, EduBridge ensures more students complete their education, acquire necessary skills, and join Rwanda's growing workforce. This not only reduces reliance on social support systems but also boosts workforce productivity, aligning with Rwanda’s economic aspirations.

        ### Conclusion and Call to Action:
        EduBridge is more than just a platform—it's a comprehensive, AI-powered solution designed to reduce dropout rates and build a more resilient and engaged education system in Rwanda. With its predictive capabilities and seamless integration with SVAS, EduBridge represents a future-proof approach to keeping students in school and ensuring educational success.

        We invite you to support EduBridge and join us in building a future where every Rwandan student has the opportunity to complete their education, fulfill their potential, and contribute to a thriving Rwanda. Together, we can make quality education a reality for Rwandan youth.
    """)

with tabs[1]:
    overview_tabs = st.tabs(["Single Prediction", "Batch Prediction"])

    with overview_tabs[0]:
        st.subheader("Enter Student Information for Single Prediction")

        student_name = st.text_input("Student Name (Optional)")
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Age: Explanation"):
                st.write("The student's age in years. Student age must be between 4 and 30.")
            age = st.number_input("Age", min_value=4, max_value=30)

        with col2:
            with st.expander("Gender: Explanation"):
                st.write("The student's gender. Select Male or Female.")
            gender = st.selectbox("Gender", options=["Male", "Female"])

        with col1:
            with st.expander("Disability Status: Explanation"):
                st.write("Indicates whether the student has a disability. Options include No Disability, Physical, Mental, or Learning.")
            disability_status = st.selectbox("Disability Status", options=["No Disability", "Physical", "Mental", "Learning"])

        with col2:
            with st.expander("School Category: Explanation"):
                st.write("The type of school the student is attending. Options include Primary or Secondary.")
            school_category = st.selectbox("School Category", options=["Primary", "Secondary"])

        with col1:
            with st.expander("Current Class: Explanation"):
                st.write("The class the student is currently enrolled in, depending on the school category.")
            if school_category == "Primary":
                current_class = st.selectbox("Current Class", options=["P1", "P2", "P3", "P4", "P5", "P6"])
            else:
                current_class = st.selectbox("Current Class", options=["S1", "S2", "S3", "S4", "S5", "S6"])

        with col2:
            with st.expander("Orphan Status: Explanation"):
                st.write("Indicates the orphan status of the student. Options are No Parents, Single (one parent), or Double (both parents).")
            orphan_status = st.selectbox("Orphan Status", options=["No Parents", "Single", "Double"])

        with col1:
            with st.expander("Household Size: Explanation"):
                st.write("The number of people in the student's household. Must be between 1 and 15.")
            household_size = st.number_input("Household Size (Number of family members)", min_value=1, max_value=15)

        with col2:
            with st.expander("Family Income Level: Explanation"):
                st.write("The income level of the student's family. Options include Very Low, Impoverished, Low, Middle, or High.")
            family_income_bracket = st.selectbox("Family Income Level", options=["Very Low", "Impoverished", "Low", "Middle", "High"])
        
        with col1:
            with st.expander("Parental Employment Status: Explanation"):
                st.write("Employment status of the student's parent or guardian. Options include Unemployed, Temporary Work, Full-Time, Part-Time, or Self-Employed.")
            parental_employment_status = st.selectbox("Parental Employment Status", options=["Unemployed", "Temporary Work", "Full-Time", "Part-Time", "Self-Employed"])

        with col2:
            with st.expander("Parental Education Level: Explanation"):
                st.write("The highest education level attained by the student's parents. Options include Non Schooled, Primary, Secondary, or Tertiary.")
            parental_education_level = st.selectbox("Parental Education Level", options=["Non Schooled", "Primary", "Secondary", "Tertiary"])

        with col1:
            with st.expander("School Fee Payment Source: Explanation"):
                st.write("The source of payment for school fees. Options include Parents, Sponsor, or Other.")
            school_fee_payment_source = st.selectbox("School Fee Payment Source", options=["Parents", "Sponsor", "Other"])

        with col2:
            with st.expander("Attendance Rate: Explanation"):
                st.write("The percentage of days the student attended school. Must be between 0 and 100%.")
            attendance_rate = st.number_input("Attendance Rate (%)", min_value=0, max_value=100)

        with col1:
            with st.expander("Days Absent Last Semester: Explanation"):
                st.write("Number of days the student was absent in the previous semester. Must be between 0 and 60.")
            days_absent_last_semester = st.number_input("Days Absent Last Semester", min_value=0, max_value=60)

        with col2:
            with st.expander("Average Grade in Last Term: Explanation"):
                st.write("The student's average grade in the last term, represented as a percentage. Must be between 0 and 100%.")
            average_grade = st.number_input("Average Grade in Last Term (%)", min_value=0, max_value=100)

        with col1:
            with st.expander("Extracurricular Activities Participation: Explanation"):
                st.write("Number of extracurricular activities the student participates in. Must be between 0 and 5.")
            activities_participation = st.number_input("Number of Extracurricular activities", min_value=0, max_value=5)

        with col2:
            with st.expander("Behavioral Infractions: Explanation"):
                st.write("The number of behavioral issues or infractions recorded for the student. Must be between 0 and 10.")
            behavioral_infractions = st.number_input("Behavioral Infractions", min_value=0, max_value=10)

        with col1:
            with st.expander("Suspensions: Explanation"):
                st.write("The number of times the student was suspended. Must be between 0 and 5.")
            suspensions = st.number_input("Suspensions", min_value=0, max_value=5)

        with col2:
            with st.expander("Distance to School: Explanation"):
                st.write("The distance from the student's home to the school, measured in meters. Must be between 0 and 10,000 meters.")
            distance_to_school = st.number_input("Distance to School (m)", min_value=0.0, max_value=10000.0)

        with col1:
            with st.expander("Transportation Mean: Explanation"):
                st.write("The mode of transportation used to reach the school. Options include Foot, Public Transport, Bicycle, or Car.")
            transportation_mean = st.selectbox("Transportation Mean", options=["Foot", "Public Transport", "Bicycle", "Car"])

        with col2:
            with st.expander("Transportation Time: Explanation"):
                st.write("Time taken by the student to reach the school, measured in minutes. Must be between 0 and 120 minutes.")
            transportation_time = st.number_input("Transportation Time (minutes)", min_value=0, max_value=120)

        with col1:
            with st.expander("Repetitions in Class: Explanation"):
                st.write("Number of times the student has repeated a class. Must be between 0 and 3.")
            repetitions_in_class = st.number_input("Repetitions in Class", min_value=0, max_value=3)

        with col2:
            with st.expander("Previous Dropout Count: Explanation"):
                st.write("The number of times the student has previously dropped out of school. Must be between 0 and 5.")
            previous_dropout_count = st.number_input("Previous Dropout Count", min_value=0, max_value=5)


        def feature_explanations(feature, shap_value):
            explanations = {
                "attendance_rate": "A lower attendance rate suggests a higher likelihood of dropout.",
                "behavioral_infractions": "More infractions can indicate disengagement or disciplinary issues.",
                "activities_participation": "Lower participation in activities can reflect a lack of engagement.",
                "repetitions_in_class": "More repetitions suggest academic struggles.",
                "family_income_bracket_Low": "Lower family income can be a risk factor.",
                "suspensions": "Higher suspensions may correlate with dropout risk.",
                "average_grade": "Lower grades may indicate academic difficulty.",
                "school_category_Secondary": "Secondary level students sometimes show higher dropout rates.",
            }

            explanation = explanations.get(feature, "")
            direction = "increases" if shap_value > 0 else "decreases"
            return f"{explanation} This feature {direction} the dropout risk."

        # Function to predict dropout and analyze feature contributions with enhanced explanations
        def analyze_student_dropout_advanced(student_data, model, preprocessor, explainer, top_n_features=21, expand=False):
            # Transform the student data
            student_processed = preprocessor.transform(student_data)

            # Predict the dropout probability and status
            dropout_probability = model.predict_proba(student_processed)[:, 1][0]
            dropout_prediction = model.predict(student_processed)[0]

            risk_status = "High Risk" if dropout_probability >= 0.3 else "Low Risk"

            # Get feature names
            numerical_features = preprocessor.transformers_[0][2]
            categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
            all_features = list(numerical_features) + list(categorical_features)


            # Create DataFrame for the processed student data
            student_processed_df = pd.DataFrame(student_processed, columns=all_features)

            # Calculate SHAP values
            shap_values = explainer(student_processed_df)
            shap_values_df = pd.DataFrame(shap_values.values[0], index=all_features, columns=["SHAP Value"])
            shap_values_df["Impact"] = shap_values_df["SHAP Value"].abs()


            # Determine the number of features to show
            num_features = top_n_features if not expand else len(shap_values_df)

            # Get top contributing features
            top_features = shap_values_df.sort_values(by='Impact', ascending=False).head(num_features)

            feature_name_mapping = {
                "age": "Age",
                "gender": "Gender",
                "school_category": "School Category",
                "attendance_rate": "Attendance Rate",
                "days_absent_last_semester": "Days Absent Last Semester",
                "average_grade": "Average Grade",
                "family_income_bracket": "Family Income",
                "household_size": "Household Size",
                "activities_participation": "Participation in Activities",
                "behavioral_infractions": "Behavioral Infractions",
                "suspensions": "Suspensions",
                "distance_to_school": "Distance to School",
                "transportation_mean": "Transportation Mean",
                "transportation_time": "Transportation Time",
                "repetitions_in_class": "Repetitions in Class",
                "previous_dropout_count": "Previous Dropout Count",
                "orphan_status": "Orphan Status",
                "disability_status": "Disability Status",
                "current_class": "Current Class",
                "parental_employment_status": "Parental Employment Status",
                "school_fee_payment_source": "School Fee Payment Source",
                "parental_education_level": "Parental Education Level",
                "parental_education_level": "Parental Education Level",
            }

            top_features.index = top_features.index.map(lambda x: feature_name_mapping.get(x, x))

            # Generate explanations and calculate importance percentage
            summary_table = top_features.copy()
            summary_table["Impact Direction"] = summary_table["SHAP Value"].apply(lambda x: "Increases" if x > 0 else "Decreases")
            total_impact = summary_table["Impact"].sum()
            summary_table["Importance (%)"] = (summary_table["Impact"] / total_impact) * 100
            summary_table["Feature Explanation"] = summary_table.apply(lambda row: feature_explanations(row.name, row["SHAP Value"]), axis=1)

            return dropout_probability, risk_status, summary_table


        if st.button("Predict Dropout Risk"):
            with st.spinner("Predicting..."):

                student_data = pd.DataFrame([{
                    "age": age,
                    "gender": gender,
                    "school_category": school_category,
                    "attendance_rate": attendance_rate,
                    "days_absent_last_semester": days_absent_last_semester,
                    "average_grade": average_grade,
                    "family_income_bracket": family_income_bracket,
                    "behavioral_infractions": behavioral_infractions,
                    "suspensions": suspensions,
                    "activities_participation": activities_participation,
                    "school_fee_payment_source": school_fee_payment_source,
                    "parental_education_level": parental_education_level,
                    "parental_employment_status": parental_employment_status,
                    "current_class": current_class,
                    "household_size": household_size,
                    "orphan_status": orphan_status,
                    "disability_status": disability_status,
                    "distance_to_school": distance_to_school,
                    "transportation_mean": transportation_mean,
                    "transportation_time": transportation_time,
                    "repetitions_in_class": repetitions_in_class,
                    "previous_dropout_count": previous_dropout_count
                }])
                
                dropout_probability, risk_status, summary_table = analyze_student_dropout_advanced(
                    student_data=student_data,
                    model=loaded_model,
                    preprocessor=loaded_preprocessor,
                    explainer=explainer
                )

                student_name_display = f"for {student_name}" if student_name else ""

                with st.expander(f"Prediction Results {student_name_display}"):
                    st.write(f"Predicted Dropout Probability: {dropout_probability * 100:.4f}%")
                    st.write(f"Risk Status: {risk_status}")

                    # Split features into "Increases" and "Decreases" categories
                    increasing_risk = summary_table[summary_table["Impact Direction"] == "Increases"]
                    decreasing_risk = summary_table[summary_table["Impact Direction"] == "Decreases"]

                    st.subheader("Top Contributing Factors")
                    st.write("### Features Increasing Dropout Risk")
                    st.dataframe(increasing_risk.style.background_gradient(cmap="Reds", subset=["Importance (%)"]))

                    st.write("### Features Decreasing Dropout Risk")
                    st.dataframe(decreasing_risk.style.background_gradient(cmap="Greens", subset=["Importance (%)"]))


                    shap_values_df = summary_table
                    # Get the explanation from Mistral
                    llm_explanation = get_llm_explanation(
                        shap_values_df=shap_values_df,
                        top_features=increasing_risk, 
                        dropout_probability=f"{dropout_probability * 100:.4f}",
                        risk_status=risk_status,
                    )

                    st.write(llm_explanation)

                    # Feature importance bar chart
                    st.write("### Feature Importance Visualization")

                    # Features Increasing Dropout Risk
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        x="Importance (%)",
                        y="index",  # Use the index column for labels
                        data=increasing_risk.reset_index(),  # Reset index to access feature names
                        hue="index",  # Assign index (feature names) to hue for color differentiation
                        palette=sns.light_palette("red", n_colors=len(increasing_risk), reverse=True),
                        ax=ax
                    )
                    ax.set_title("Features Increasing Dropout Risk")
                    ax.set_xlabel("Importance (%)")
                    ax.set_ylabel("Feature")
                    st.pyplot(fig)

                    # Features Decreasing Dropout Risk
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        x="Importance (%)",
                        y="index",  # Use the index column for labels
                        data=decreasing_risk.reset_index(),  # Reset index to access feature names
                        hue="index",  # Assign index (feature names) to hue for color differentiation
                        palette=sns.light_palette("green", n_colors=len(decreasing_risk), reverse=True),
                        ax=ax
                    )
                    ax.set_title("Features Decreasing Dropout Risk")
                    ax.set_xlabel("Importance (%)")
                    ax.set_ylabel("Feature")
                    st.pyplot(fig)

    with overview_tabs[1]:
        st.subheader("Upload File for Batch Prediction")
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

        with st.expander("Required Fields in the Uploaded Dataset:", expanded=False):  
            st.write("""  
                - **age**: Age of the student.  
                - **gender**: Gender of the student (e.g., Male/Female).  
                - **school_category**: Category of the school (e.g., Public/Private).  
                - **attendance_rate**: Attendance rate of the student (as a percentage).  
                - **days_absent_last_semester**: Total days the student was absent in the last semester.  
                - **average_grade**: Average academic grade of the student.  
                - **family_income_bracket**: Income bracket of the family (e.g., Low/Medium/High).  
                - **behavioral_infractions**: Number of behavioral infractions reported.  
                - **suspensions**: Number of suspensions.  
                - **activities_participation**: Participation in extracurricular activities.  
                - **transportation_time**: Time taken to travel to school (in minutes).  
                - **school_fee_payment_source**: Source of school fee payments (e.g., Parents/Sponsors).  
                - **repetitions_in_class**: Number of times the student has repeated a class.  
                - **current_class**: Current grade/class of the student.  
                - **household_size**: Number of people in the household.  
                - **orphan_status**: Whether the student is an orphan or not.  
                - **disability_status**: Any reported disabilities (if applicable).  
                - **distance_to_school**: Distance to school (in kilometers).  
                - **parental_employment_status**: Employment status of parents (e.g., Employed/Unemployed).  
                - **previous_dropout_count**: Number of previous dropouts by the student.  
            """)
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.write("**Uploaded Data**")
            st.write(data.head())

            required_columns = [
                "age", "gender", "school_category", "attendance_rate",
                "days_absent_last_semester", "average_grade", "family_income_bracket",
                "behavioral_infractions", "suspensions", "activities_participation",
                "transportation_time", "school_fee_payment_source", "repetitions_in_class",
                "current_class", "household_size", "orphan_status", "disability_status",
                "distance_to_school", "parental_employment_status", "previous_dropout_count"
            ]

            # Check if required columns exist in the uploaded file
            if set(required_columns).issubset(data.columns):
                all_students_results = []
                for i, row in data.iterrows():
                    student_data = pd.DataFrame([row])

                    # Analyze dropout risk using the custom function
                    dropout_probability, risk_status, summary_table = analyze_student_dropout_advanced(
                        student_data, loaded_model, loaded_preprocessor, explainer
                    )

                    # Add predictions to the result list
                    student_result = row.copy()
                    student_result["Dropout Probability (%)"] = round(dropout_probability * 100, 2)
                    student_result["Risk Level"] = risk_status
                    all_students_results.append(student_result)

                    st.write(f"**Student {i + 1}:**")
                    st.write(f"Predicted Dropout Probability: {dropout_probability * 100:.8f}%")
                    st.write(f"Risk Status: {risk_status}")

                    # Highlight features based on their impact direction
                    def highlight_features(row):
                        if row["Impact Direction"] == "Increases":
                            return ["background-color: red" if col == "Impact Direction" else "" for col in row.index]
                        elif row["Impact Direction"] == "Decreases":
                            return ["background-color: green" if col == "Impact Direction" else "" for col in row.index]
                        else:
                            return [""] * len(row)

                    # Ensure the Impact Direction column exists in the summary table
                    st.write("### Top Contributing Factors")
                    if "Impact Direction" in summary_table.columns:
                        # Apply styling to the entire dataframe
                        styled_summary_table = summary_table.style.apply(highlight_features, axis=1)
                        st.dataframe(styled_summary_table)
                    else:
                        st.warning("The 'Impact Direction' column is missing in the summary table.")


           
                # General table for all students
                results_df = pd.DataFrame(all_students_results)
                st.write("**General Table with Predicted Probabilities and Risk Levels**")
                st.write(results_df[required_columns + ["Dropout Probability (%)", "Risk Level"]])

                # Download results
                st.download_button(
                    label="Download Results as CSV",
                    data=results_df.to_csv(index=False),
                    file_name="dropout_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("The uploaded file does not have the required columns.")

with tabs[2]:
    talk_to_assistant()

with tabs[3]:
    developers_tab()

with tabs[4]:
    contact_us_tab()