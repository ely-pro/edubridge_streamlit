# EduBridge | Insight Today, Impact Tomorrow

EduBridge is a transformative AI-driven platform designed to tackle one of Rwanda’s biggest educational challenges: student dropout prevention. By leveraging advanced data analytics and machine learning, EduBridge predicts and prevents student dropouts, providing early intervention to keep students on track and engaged in their education.

Rwanda has made remarkable strides in education, with achievements like Universal Primary Education and net enrollment rates consistently above 97.5%. Despite these successes, the country still faces challenges with high dropout rates, particularly at critical transition points, such as from primary to secondary school. This is where EduBridge comes in — helping educators and policymakers identify students at risk of dropping out before it's too late.

## Table of Contents
1. [Introduction](#introduction)
2. [Technologies](#technologies)
3. [Getting Started](#getting-started)
   - [Clone the Repository](#clone-the-repository)
   - [Install Dependencies](#install-dependencies)
   - [Running the Project](#running-the-project)

## Introduction

EduBridge is designed to help educational institutions and policymakers in Rwanda identify students who are at risk of dropping out before it's too late. The platform leverages data analytics and machine learning to predict dropouts, and provides actionable insights for intervention strategies. By integrating predictive models and real-time data analytics, EduBridge empowers schools to create a supportive environment that encourages students to remain engaged in their education.

## Technologies

- **Python**: Programming Language
- **Streamlit**: Web Framework for creating interactive apps
- **Pandas**: Data Manipulation and Analysis
- **Scikit-learn**: Machine Learning Library
- **Matplotlib** / **Seaborn**: Data Visualization Libraries

## Getting Started

Follow these steps to set up and run EduBridge on your local machine.

### Clone the Repository

To get a copy of this project, follow these steps:

1. Make sure **Git** is installed on your machine. If not, download and install it from [here](https://git-scm.com/).
2. Open a terminal or command prompt and clone the repository:

   ```bash
   git clone https://github.com/ely-pro/edubridge_streamlit.git
   ```

3. Navigate to the project directory:

   ```bash
   cd edubridge_streamlit
   ```

### Install Dependencies

To install the necessary dependencies, follow these steps:

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:

   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install the required Python packages**:

   If you have a `requirements.txt` file in your project, run:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

Once you have installed the dependencies, follow these steps to run the EduBridge project:

1. Open a terminal window.
2. Navigate to the project directory if you're not already there.
3. Run the following command to start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. The Streamlit server will start, and you should see an output in the terminal similar to this:

   ```
   You can view your application in your browser by visiting the following URL:
   http://localhost:8501
   ```

   Open the provided URL in your web browser to interact with EduBridge.
