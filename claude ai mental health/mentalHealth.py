# Mental Health Medical Help Prediction - Complete ML Project
# This project predicts whether a person needs medical help based on demographic and lifestyle data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class MentalHealthPredictor:
    """
    A complete machine learning pipeline for predicting mental health medical needs
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        self.feature_columns = None
        self.results = {}
        
    def load_and_explore_data(self, csv_path):
        """
        Load the dataset and perform initial exploration
        """
        print("=== LOADING AND EXPLORING DATA ===")
        
        # Load the dataset
        self.df = pd.read_csv(csv_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Display basic information
        print("\n--- Dataset Info ---")
        print(self.df.info())
        
        print("\n--- Missing Values ---")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\n--- Target Variable Distribution ---")
        print("Mental Health Condition:")
        print(self.df['Mental_Health_Condition'].value_counts())
        print("\nSeverity:")
        print(self.df['Severity'].value_counts())
        
        return self.df
    
    def create_target_variable(self):
        """
        Create the target variable: Needs medical help = Mental_Health_Condition is Yes AND Severity is Medium or High
        """
        print("\n=== CREATING TARGET VARIABLE ===")
        
        # Create the target variable
        self.df['Needs_Medical_Help'] = (
            (self.df['Mental_Health_Condition'] == 'Yes') & 
            (self.df['Severity'].isin(['Medium', 'High']))
        ).astype(int)
        
        # Display target distribution
        target_distribution = self.df['Needs_Medical_Help'].value_counts()
        print("Target Variable Distribution:")
        print(f"No medical help needed (0): {target_distribution[0]} ({target_distribution[0]/len(self.df)*100:.1f}%)")
        print(f"Medical help needed (1): {target_distribution[1]} ({target_distribution[1]/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def preprocess_data(self):
        """
        Comprehensive data preprocessing including handling missing values, encoding, and scaling
        """
        print("\n=== PREPROCESSING DATA ===")
        
        # Remove unnecessary columns for prediction (keeping original target columns for reference)
        features_to_drop = ['User_ID', 'Mental_Health_Condition', 'Severity']
        X = self.df.drop(columns=features_to_drop + ['Needs_Medical_Help'])
        y = self.df['Needs_Medical_Help']
        
        # Store feature column names
        self.feature_columns = X.columns.tolist()
        print(f"Features used for prediction: {self.feature_columns}")
        
        # Handle missing values (fill with mode for categorical, median for numerical)
        print("\n--- Handling Missing Values ---")
        
        # Numerical columns
        numerical_cols = ['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage']
        
        # Categorical columns
        categorical_cols = ['Gender', 'Occupation', 'Country', 'Consultation_History', 'Stress_Level', 
                           'Diet_Quality', 'Smoking_Habit', 'Alcohol_Consumption', 'Medication_Usage']
        
        # Fill missing values
        for col in numerical_cols:
            if col in X.columns and X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
                print(f"Filled {col} missing values with median: {X[col].median():.2f}")
        
        for col in categorical_cols:
            if col in X.columns and X[col].isnull().sum() > 0:
                mode_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                X[col].fillna(mode_value, inplace=True)
                print(f"Filled {col} missing values with mode: {mode_value}")
        
        # Create preprocessing pipeline
        print("\n--- Creating Preprocessing Pipeline ---")
        
        # Define preprocessing for numerical and categorical features
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        print("Preprocessing pipeline created successfully")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets
        """
        print(f"\n=== SPLITTING DATA (Train: {1-test_size:.0%}, Test: {test_size:.0%}) ===")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        print(f"Training set positive class: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        print(f"Testing set positive class: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train Random Forest model
        """
        print("\n=== TRAINING MODEL ===")
        print("Training Random Forest Classifier...")
        
        # Define Random Forest model with hyperparameter tuning
        model_config = {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }
        }
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model_config['model'])
        ])
        
        # Perform grid search with cross-validation
        print("Optimizing hyperparameters...")
        grid_search = GridSearchCV(
            pipeline, 
            model_config['params'], 
            cv=5, 
            scoring='accuracy',  # Use accuracy as requested
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Store the best model
        self.best_model = grid_search.best_estimator_
        self.models['Random Forest'] = grid_search.best_estimator_
        
        print(f"Training completed!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Cross-validation accuracy: {grid_search.best_score_:.4f}")
        
        return self.best_model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model and display accuracy
        """
        print("\n=== EVALUATING MODEL ===")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy on Test Set: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy
    
    def save_model(self, filename='mental_health_predictor.pkl'):
        """
        Save the trained model to a file
        """
        print(f"\n=== SAVING MODEL ===")
        
        # Save the model along with feature columns
        model_data = {
            'model': self.best_model,
            'feature_columns': self.feature_columns,
            'preprocessor': self.preprocessor
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as: {filename}")
        
        return filename
    
    @staticmethod
    def load_model(filename='mental_health_predictor.pkl'):
        """
        Load a trained model from file
        """
        model_data = joblib.load(filename)
        return model_data['model'], model_data['feature_columns'], model_data['preprocessor']
    
    def predict_single_person(self, person_data, model_filename=None):
        """
        Predict whether a single person needs medical help
        
        Args:
            person_data: Dictionary with person's information
            model_filename: Path to saved model file (optional)
        
        Returns:
            Dictionary with prediction results
        """
        
        # Load model if filename provided
        if model_filename:
            model, feature_columns, preprocessor = self.load_model(model_filename)
        else:
            model = self.best_model
            feature_columns = self.feature_columns
        
        # Convert person data to DataFrame
        person_df = pd.DataFrame([person_data])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in person_df.columns:
                person_df[col] = np.nan  # Will be handled by preprocessing
        
        # Reorder columns to match training data
        person_df = person_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(person_df)[0]
        probability = model.predict_proba(person_df)[0]
        
        # Create result dictionary
        result = {
            'needs_medical_help': bool(prediction),
            'probability_no_help': float(probability[0]),
            'probability_needs_help': float(probability[1]),
            'confidence': float(max(probability)),
            'recommendation': self._get_recommendation(prediction, probability)
        }
        
        return result
    
    def _get_recommendation(self, prediction, probability):
        """
        Generate a human-readable recommendation based on prediction
        """
        confidence = max(probability)
        needs_help = bool(prediction)
        
        if needs_help:
            if confidence > 0.8:
                return "Strong recommendation to seek professional mental health support immediately."
            elif confidence > 0.6:
                return "Recommended to consult with a mental health professional."
            else:
                return "Consider speaking with a healthcare provider about your mental health."
        else:
            if confidence > 0.8:
                return "Low risk indicators. Continue maintaining good mental health practices."
            else:
                return "Mixed indicators. Monitor your mental health and consider consultation if symptoms persist."

def run_complete_pipeline(csv_file_path="mental_health.csv"):
    """
    Run the complete machine learning pipeline
    """
    print("🧠 MENTAL HEALTH MEDICAL HELP PREDICTION SYSTEM 🧠")
    print("=" * 60)
    
    # Initialize the predictor
    predictor = MentalHealthPredictor()
    
    # Step 1: Load and explore data
    df = predictor.load_and_explore_data(csv_file_path)
    
    # Step 2: Create target variable
    df = predictor.create_target_variable()
    
    # Step 3: Preprocess data
    X, y = predictor.preprocess_data()
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Step 5: Train model
    model = predictor.train_model(X_train, y_train)
    
    # Step 6: Evaluate model
    accuracy = predictor.evaluate_model(X_test, y_test)
    
    # Step 7: Save the model
    model_filename = predictor.save_model()
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📁 Model saved as: {model_filename}")
    print(f"🎯 Final Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return predictor

def get_user_input_interactive():
    """
    Interactive function to collect user data through command-line input
    """
    print("\n🏥 MENTAL HEALTH ASSESSMENT - DATA INPUT")
    print("=" * 50)
    print("Please provide the following information:\n")
    
    person_data = {}
    
    # Define input prompts and validation
    input_config = {
        'Age': {
            'prompt': "Enter your age (18-100): ",
            'type': 'int',
            'validation': lambda x: 18 <= x <= 100
        },
        'Gender': {
            'prompt': "Enter your gender (Male/Female/Other): ",
            'type': 'str',
            'options': ['Male', 'Female', 'Other']
        },
        'Occupation': {
            'prompt': "Enter your occupation (e.g., Teacher, Engineer, Student, etc.): ",
            'type': 'str'
        },
        'Country': {
            'prompt': "Enter your country: ",
            'type': 'str'
        },
        'Consultation_History': {
            'prompt': "Have you consulted a mental health professional before? (Yes/No): ",
            'type': 'str',
            'options': ['Yes', 'No']
        },
        'Stress_Level': {
            'prompt': "How would you rate your current stress level? (Low/Medium/High): ",
            'type': 'str',
            'options': ['Low', 'Medium', 'High']
        },
        'Sleep_Hours': {
            'prompt': "How many hours do you sleep per night on average? (1-12): ",
            'type': 'float',
            'validation': lambda x: 1 <= x <= 12
        },
        'Work_Hours': {
            'prompt': "How many hours do you work per week? (0-100): ",
            'type': 'float',
            'validation': lambda x: 0 <= x <= 100
        },
        'Physical_Activity_Hours': {
            'prompt': "How many hours of physical activity do you do per week? (0-20): ",
            'type': 'float',
            'validation': lambda x: 0 <= x <= 20
        },
        'Social_Media_Usage': {
            'prompt': "How many hours per day do you spend on social media? (0-24): ",
            'type': 'float',
            'validation': lambda x: 0 <= x <= 24
        },
        'Diet_Quality': {
            'prompt': "How would you rate your diet quality? (Healthy/Average/Unhealthy): ",
            'type': 'str',
            'options': ['Healthy', 'Average', 'Unhealthy']
        },
        'Smoking_Habit': {
            'prompt': "What is your smoking habit? (Non-Smoker/Regular Smoker/Heavy Smoker): ",
            'type': 'str',
            'options': ['Non-Smoker', 'Regular Smoker', 'Heavy Smoker']
        },
        'Alcohol_Consumption': {
            'prompt': "What is your alcohol consumption pattern? (Non-Drinker/Social Drinker/Regular Drinker): ",
            'type': 'str',
            'options': ['Non-Drinker', 'Social Drinker', 'Regular Drinker']
        },
        'Medication_Usage': {
            'prompt': "Are you currently taking any medications? (Yes/No): ",
            'type': 'str',
            'options': ['Yes', 'No']
        }
    }
    
    # Collect input for each field
    for field, config in input_config.items():
        while True:
            try:
                # Show options if available
                if 'options' in config:
                    print(f"Options: {', '.join(config['options'])}")
                
                # Get user input
                user_input = input(config['prompt']).strip()
                
                # Convert to appropriate type
                if config['type'] == 'int':
                    value = int(user_input)
                elif config['type'] == 'float':
                    value = float(user_input)
                else:
                    value = user_input
                
                # Validate input
                if 'options' in config and value not in config['options']:
                    print(f"❌ Please choose from: {', '.join(config['options'])}")
                    continue
                
                if 'validation' in config and not config['validation'](value):
                    print("❌ Invalid input. Please try again.")
                    continue
                
                # Store valid input
                person_data[field] = value
                print(f"✅ {field}: {value}\n")
                break
                
            except ValueError:
                print(f"❌ Please enter a valid {config['type']} value.")
            except KeyboardInterrupt:
                print("\n\n❌ Input cancelled by user.")
                return None
    
    return person_data

def get_user_input_guided():
    """
    Guided input with more detailed explanations and examples
    """
    print("\n🧠 COMPREHENSIVE MENTAL HEALTH ASSESSMENT")
    print("=" * 55)
    print("This assessment will help evaluate if you might benefit from professional mental health support.")
    print("Please answer all questions honestly. Your privacy is important - this data is not stored.\n")
    
    person_data = {}
    
    # Demographic Information
    print("📋 DEMOGRAPHIC INFORMATION")
    print("-" * 25)
    
    # Age
    while True:
        try:
            age = int(input("What is your age? (18-100): "))
            if 18 <= age <= 100:
                person_data['Age'] = age
                break
            else:
                print("Please enter an age between 18 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Gender
    print("\nGender options: Male, Female, Other")
    while True:
        gender = input("What is your gender? ").strip()
        if gender in ['Male', 'Female', 'Other']:
            person_data['Gender'] = gender
            break
        else:
            print("Please enter Male, Female, or Other.")
    
    # Occupation
    print("\nExamples: Teacher, Engineer, Student, Healthcare Worker, Unemployed, Retired, etc.")
    person_data['Occupation'] = input("What is your occupation? ").strip()
    
    # Country
    person_data['Country'] = input("What country are you from? ").strip()
    
    # Mental Health History
    print("\n🩺 MENTAL HEALTH HISTORY")
    print("-" * 25)
    
    print("Have you ever consulted with a mental health professional (therapist, psychologist, psychiatrist)?")
    while True:
        consultation = input("Consultation History (Yes/No): ").strip()
        if consultation in ['Yes', 'No']:
            person_data['Consultation_History'] = consultation
            break
        else:
            print("Please enter Yes or No.")
    
    # Current medication
    print("Are you currently taking any medications (including antidepressants, anti-anxiety, etc.)?")
    while True:
        medication = input("Current Medication Usage (Yes/No): ").strip()
        if medication in ['Yes', 'No']:
            person_data['Medication_Usage'] = medication
            break
        else:
            print("Please enter Yes or No.")
    
    # Lifestyle Factors
    print("\n🏃 LIFESTYLE FACTORS")
    print("-" * 20)
    
    # Stress Level
    print("How would you rate your current overall stress level?")
    print("Low: Minimal stress, feeling relaxed most of the time")
    print("Medium: Moderate stress, manageable but noticeable")
    print("High: Significant stress, feeling overwhelmed frequently")
    while True:
        stress = input("Stress Level (Low/Medium/High): ").strip()
        if stress in ['Low', 'Medium', 'High']:
            person_data['Stress_Level'] = stress
            break
        else:
            print("Please enter Low, Medium, or High.")
    
    # Sleep
    while True:
        try:
            sleep = float(input("How many hours do you sleep per night on average? (1-12): "))
            if 1 <= sleep <= 12:
                person_data['Sleep_Hours'] = sleep
                break
            else:
                print("Please enter a number between 1 and 12.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Work Hours
    while True:
        try:
            work = float(input("How many hours do you work per week? (0-100): "))
            if 0 <= work <= 100:
                person_data['Work_Hours'] = work
                break
            else:
                print("Please enter a number between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Physical Activity
    while True:
        try:
            activity = float(input("How many hours of physical activity/exercise do you do per week? (0-20): "))
            if 0 <= activity <= 20:
                person_data['Physical_Activity_Hours'] = activity
                break
            else:
                print("Please enter a number between 0 and 20.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Social Media
    while True:
        try:
            social_media = float(input("How many hours per day do you spend on social media? (0-24): "))
            if 0 <= social_media <= 24:
                person_data['Social_Media_Usage'] = social_media
                break
            else:
                print("Please enter a number between 0 and 24.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Diet Quality
    print("\nHow would you rate your overall diet quality?")
    print("Healthy: Balanced diet with fruits, vegetables, regular meals")
    print("Average: Somewhat balanced but with some unhealthy choices")
    print("Unhealthy: Frequent fast food, irregular meals, poor nutrition")
    while True:
        diet = input("Diet Quality (Healthy/Average/Unhealthy): ").strip()
        if diet in ['Healthy', 'Average', 'Unhealthy']:
            person_data['Diet_Quality'] = diet
            break
        else:
            print("Please enter Healthy, Average, or Unhealthy.")
    
    # Smoking
    print("\nWhat best describes your smoking habits?")
    print("Non-Smoker: Don't smoke at all")
    print("Regular Smoker: Smoke occasionally or regularly")
    print("Heavy Smoker: Smoke heavily/frequently")
    while True:
        smoking = input("Smoking Habit (Non-Smoker/Regular Smoker/Heavy Smoker): ").strip()
        if smoking in ['Non-Smoker', 'Regular Smoker', 'Heavy Smoker']:
            person_data['Smoking_Habit'] = smoking
            break
        else:
            print("Please enter Non-Smoker, Regular Smoker, or Heavy Smoker.")
    
    # Alcohol
    print("\nWhat best describes your alcohol consumption?")
    print("Non-Drinker: Don't drink alcohol")
    print("Social Drinker: Drink occasionally in social settings")
    print("Regular Drinker: Drink regularly/frequently")
    while True:
        alcohol = input("Alcohol Consumption (Non-Drinker/Social Drinker/Regular Drinker): ").strip()
        if alcohol in ['Non-Drinker', 'Social Drinker', 'Regular Drinker']:
            person_data['Alcohol_Consumption'] = alcohol
            break
        else:
            print("Please enter Non-Drinker, Social Drinker, or Regular Drinker.")
    
    return person_data

def predict_with_user_input(model_filename='mental_health_predictor.pkl', guided_input=True):
    """
    Complete function to get user input and make prediction
    
    Args:
        model_filename: Path to saved model file
        guided_input: If True, use guided input; if False, use simple interactive input
    """
    
    print("🧠 MENTAL HEALTH PREDICTION SYSTEM")
    print("=" * 40)
    
    # Check if model file exists
    try:
        import os
        if not os.path.exists(model_filename):
            print(f"❌ Model file '{model_filename}' not found!")
            print("Please train the model first using run_complete_pipeline()")
            return None
    except ImportError:
        pass
    
    # Get user input
    if guided_input:
        print("Choose input method:")
        print("1. Guided input (detailed explanations)")
        print("2. Simple input (quick entry)")
        
        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1':
                person_data = get_user_input_guided()
                break
            elif choice == '2':
                person_data = get_user_input_interactive()
                break
            else:
                print("Please enter 1 or 2.")
    else:
        person_data = get_user_input_interactive()
    
    if person_data is None:
        print("Assessment cancelled.")
        return None
    
    # Display collected data for confirmation
    print("\n📋 SUMMARY OF YOUR INPUT:")
    print("-" * 30)
    for key, value in person_data.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Confirm before prediction
    while True:
        confirm = input("\nIs this information correct? (Yes/No): ").strip()
        if confirm in ['Yes', 'Y', 'yes', 'y']:
            break
        elif confirm in ['No', 'N', 'no', 'n']:
            print("Please restart the assessment to enter correct information.")
            return None
        else:
            print("Please enter Yes or No.")
    
    # Make prediction
    try:
        predictor = MentalHealthPredictor()
        result = predictor.predict_single_person(person_data, model_filename)
        
        # Display results
        print("\n" + "=" * 50)
        print("🔍 MENTAL HEALTH ASSESSMENT RESULTS")
        print("=" * 50)
        
        # Main result
        if result['needs_medical_help']:
            print("🚨 RECOMMENDATION: SEEK PROFESSIONAL HELP")
            print(f"   Our model suggests you may benefit from professional mental health support.")
        else:
            print("✅ ASSESSMENT: LOW RISK INDICATORS")
            print(f"   Our model suggests low risk, but continue monitoring your mental health.")
        
        print(f"\n📊 CONFIDENCE LEVEL: {result['confidence']:.1%}")
        print(f"📈 PROBABILITY OF NEEDING HELP: {result['probability_needs_help']:.1%}")
        
        print(f"\n💡 DETAILED RECOMMENDATION:")
        print(f"   {result['recommendation']}")
        
        # Important disclaimers
        print("\n" + "⚠️  IMPORTANT DISCLAIMERS" + " ⚠️ ")
        print("-" * 25)
        print("• This is an AI assessment tool and NOT a medical diagnosis")
        print("• Always consult with qualified mental health professionals")
        print("• If you're in crisis, contact emergency services immediately")
        print("• Results are based on statistical patterns, not individual evaluation")
        
        # Resources
        print("\n📞 MENTAL HEALTH RESOURCES:")
        print("-" * 25)
        print("• National Suicide Prevention Lifeline: 988 (US)")
        print("• Crisis Text Line: Text HOME to 741741")
        print("• International Association for Suicide Prevention: https://iasp.info/")
        
        return result
        
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        print("Please ensure the model file exists and try again.")
        return None

def predict_new_person(person_data, model_filename='mental_health_predictor.pkl'):
    """
    Convenience function to predict for a new person using saved model (for programmatic use)
    """
    
    predictor = MentalHealthPredictor()
    result = predictor.predict_single_person(person_data, model_filename)
    
    print(f"\n🔍 PREDICTION RESULTS:")
    print("-" * 40)
    print(f"Needs Medical Help: {'YES' if result['needs_medical_help'] else 'NO'}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Probability of needing help: {result['probability_needs_help']:.1%}")
    print(f"Recommendation: {result['recommendation']}")
    
    return result

# Main execution and usage examples
if __name__ == "__main__":
    print("🧠 MENTAL HEALTH PREDICTION SYSTEM")
    print("=" * 45)
    print("\nChoose an option:")
    print("1. Train new model")
    print("2. Use existing model for assessment")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            # Train new model - assumes dataset is named "mental_health_data.csv"
            print("Training model with dataset: mental_health_data.csv")
            try:
                predictor = run_complete_pipeline()
                print("\n✅ Model training completed!")
                print("You can now use option 2 for assessments.")
            except Exception as e:
                print(f"❌ Error training model: {e}")
                print("Please ensure your CSV file is named 'mental_health_data.csv' and is in the same directory.")
        
        elif choice == '2':
            # Use existing model for assessment
            result = predict_with_user_input('mental_health_predictor.pkl', guided_input=True)
            
        elif choice == '3':
            print("👋 Thank you for using the Mental Health Prediction System!")
            print("Remember: Always consult with healthcare professionals for proper diagnosis.")
            break
            
        else:
            print("❌ Please enter a number between 1 and 3.")