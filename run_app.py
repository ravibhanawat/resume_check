#!/usr/bin/env python3
"""
Simple script to run the ATS Resume Checker application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'PyPDF2',
        'python-docx',
        'nltk',
        'spacy',
        'textstat',
        'pandas',
        'numpy',
        'scikit-learn',
        'wordcloud',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print("pip3 install -r requirements.txt")
        return False
    
    return True

def check_spacy_model():
    """Check if spaCy English model is installed"""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except OSError:
        print("spaCy English model not found.")
        print("Please install it using:")
        print("python3 -m spacy download en_core_web_sm")
        return False

def run_app():
    """Run the Streamlit application"""
    if not check_dependencies():
        sys.exit(1)
    
    if not check_spacy_model():
        sys.exit(1)
    
    print("Starting ATS Resume Checker...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ats_resume_checker.py",
            "--server.headless", "false",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    run_app()