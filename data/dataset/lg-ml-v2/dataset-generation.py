import pandas as pd
import numpy as np
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Define the number of records
num_records = 500

# Define class levels for primary and secondary education in Rwanda
primary_classes = ["P1", "P2", "P3", "P4", "P5", "P6"]
secondary_classes = ["S1", "S2", "S3", "S4", "S5", "S6"]

# New family income brackets
income_brackets = ["Very Low", "Impoverished", "Low", "Middle", "High"]

# Function to generate a single student record based on dropout status
def generate_student_record(dropout_status):
    # Determine school category and assign class level and age
    category = "Primary" if random.random() < 0.55 else "Secondary"
    if category == "Primary":
        current_class = random.choice(primary_classes)
        age = random.randint(7, 13)
    else:
        current_class = random.choice(secondary_classes)
        age = random.randint(14, 21)
    
    # Adjust dropout likelihood based on transition years and age-class compatibility
    if (current_class == "P1" and age > 9) or (current_class == "S1" and age > 15):
        dropout_status = 1
    
    if (category == "Secondary" and current_class in ["S1", "S4"]) or current_class == "P1":
        dropout_status = 1 if random.random() < 0.5 else dropout_status

    # Set dropout-related features for a realistic scenario
    if dropout_status == 1:
        # Introducing factors that could still prevent dropout
        attendance_rate = round(random.uniform(20, 50), 1)
        days_absent = random.randint(15, 40)
        average_grade = round(random.uniform(20, 50), 2)
        behavioral_infractions = random.randint(3, 7)
        suspensions = random.randint(0, 3)
        family_income_bracket = random.choice(income_brackets[:3])  # Very Low, Poverty, Low
        household_size = random.randint(6, 12)
        orphan_status = random.choices(
            ["No Parents", "Single", "Double"],
            weights=[30, 30, 40], 
            k=1
        )[0]

        disability_status = random.choices(
            ["No Disability", "Physical", "Mental", "Learning"],
            weights=[50, 15, 15, 20],
            k=1
        )[0]
        distance_to_school = random.uniform(5000, 10000)  # Higher distances for dropouts
        
        transportation_mean = random.choices(
            ["Foot", "Public Transport", "Bicycle", "Car"],
            weights=[80, 10, 5, 5],
            k=1
        )[0]
        if transportation_mean == "Foot":
            transportation_time = round(distance_to_school / 100, 1)  # Walking approx. 1 min per 100 meters
        elif transportation_mean == "Public Transport":
            transportation_time = round(distance_to_school / 300, 1)  # Public transport approx. 1 min per 300 meters
        elif transportation_mean == "Bicycle":
            transportation_time = round(distance_to_school / 250, 1)  # Biking approx. 1 min per 250 meters
        elif transportation_mean == "Car":
            transportation_time = round(distance_to_school / 500, 1)  # Car approx. 1 min per 500 meters


        parental_education_level = random.choices(
            ["Non Schooled", "Primary", "Secondary", "Tertiary"],
            weights=[40, 30, 20, 10],
            k=1
        )[0]
        parental_employment_status = random.choices(
            ["Unemployed", "Temporary Work", "Full-Time", "Part-Time", "Self-Employed"],
            weights=[40, 30, 5, 10, 15],
            k=1
        )[0]
        previous_dropout_count = random.choices(
            [0, 1],
            weights=[70, 30],
            k=1
        )[0]
    else:
        # Introducing factors that could contribute to dropout even for non-dropouts
        attendance_rate = round(random.uniform(70, 95), 1)
        days_absent = random.randint(0, 5) 
        average_grade = round(random.uniform(50, 98), 2) 
        behavioral_infractions = random.randint(0, 2)
        suspensions = random.randint(0, 1)
        family_income_bracket = random.choice(income_brackets[2:]) 
        household_size = random.randint(3, 6)
        orphan_status = random.choices(
            ["No Parents", "Single", "Double"],
            weights=[10, 15, 75], 
            k=1
        )[0]
        disability_status = random.choices(
            ["No Disability", "Physical", "Mental", "Learning"],
            weights=[85, 5, 5, 5],
            k=1
        )[0]
        distance_to_school = random.uniform(500, 3000) 
        
        # Likely transportation for non-dropout students
        transportation_mean = random.choices(
            ["Foot", "Public Transport", "Bicycle", "Car"],
            weights=[30, 30, 20, 20],
            k=1
        )[0]
        if transportation_mean == "Foot":
            transportation_time = round(distance_to_school / 100, 1)  # Walking approx. 1 min per 100 meters
        elif transportation_mean == "Public Transport":
            transportation_time = round(distance_to_school / 300, 1)  # Public transport approx. 1 min per 300 meters
        elif transportation_mean == "Bicycle":
            transportation_time = round(distance_to_school / 250, 1)  # Biking approx. 1 min per 250 meters
        elif transportation_mean == "Car":
            transportation_time = round(distance_to_school / 500, 1)  # Car approx. 1 min per 500 meters

        parental_education_level = random.choices(
            ["Non Schooled", "Primary", "Secondary", "Tertiary"],
            weights=[10, 20, 30, 40],
            k=1
        )[0]
        parental_employment_status = random.choices(
            ["Unemployed", "Temporary Work", "Full-Time", "Part-Time", "Self-Employed"],
            weights=[5, 10, 35, 20, 30],
            k=1
        )[0]
        previous_dropout_count = random.choices(
            [0, 1],
            weights=[95, 5],
            k=1
        )[0]


    # Create a record with refined details to reflect a realistic student profile
    record = {
        "age": age,
        "gender": random.choice(["Male", "Female"]),
        "disability_status": disability_status,
        "school_category": category,
        "attendance_rate": attendance_rate,
        "days_absent_last_semester": days_absent,
        "average_grade": average_grade,
        "household_size": household_size,
        "orphan_status": orphan_status,
        "family_income_bracket": family_income_bracket,
        "parental_education_level": parental_education_level,
        "parental_employment_status": parental_employment_status,
        "school_fee_payment_source": random.choice(["Parents", "Sponsor", "Other"]),
        "behavioral_infractions": behavioral_infractions,
        "suspensions": suspensions,
        "distance_to_school": distance_to_school,
        "transportation_time": transportation_time,
        "transportation_mean": transportation_mean,
        "activities_participation": random.randint(0, 2) if dropout_status == 1 else random.randint(3, 5),
        "repetitions_in_class": random.randint(1, 3) if dropout_status == 1 else random.randint(0, 1),
        "current_class": current_class,
        "previous_dropout_count": previous_dropout_count,
        "dropout_status": dropout_status
    }

    return record

# Function to generate the full dataset with a realistic dropout rate
def generate_dataset(num_records):
    data = []
    
    # Set dropout rate to around 20%
    num_dropout = int(num_records * 0.4)
    num_stay = num_records - num_dropout

    # Generate dropout records
    for _ in range(num_dropout):
        record = generate_student_record(dropout_status=1)
        data.append(record)

    # Generate staying-in-school records
    for _ in range(num_stay):
        record = generate_student_record(dropout_status=0)
        data.append(record)

    # Shuffle the dataset to mix dropout and non-dropout records
    random.shuffle(data)

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv('students_dataset.csv', index=False)
    return df

# Generate and display the dataset
df = generate_dataset(num_records)
print(df.head())
