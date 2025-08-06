"""Test script for Groq API integration - non-interactive."""

import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def test_groq_integration():
    print("Testing Groq API integration...")
    
    # Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment")
        return
    
    print("SUCCESS: GROQ_API_KEY found")
    
    # Initialize Groq client
    try:
        groq_client = Groq(api_key=api_key)
        print("SUCCESS: Groq client initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize Groq client: {e}")
        return
    
    # Test direct Groq API
    print("\nTesting direct Groq API call:")
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=10,
            temperature=0
        )
        print(f"SUCCESS: Groq response: {completion.choices[0].message.content}")
    except Exception as e:
        print(f"ERROR: Groq API call failed: {e}")
        return
    
    # Test with math problems
    test_problems = [
        "What is 25 + 37?",
        "John has 15 apples. He buys 8 more and eats 3. How many apples does he have?",
        "A train travels 180 miles in 3 hours. What is its speed?",
    ]
    
    print("\nTesting Groq with math problems:")
    print("-" * 50)
    
    for problem in test_problems:
        print(f"\nProblem: {problem}")
        try:
            completion = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful math tutor. Give concise answers."},
                    {"role": "user", "content": problem}
                ],
                max_tokens=50,
                temperature=0
            )
            answer = completion.choices[0].message.content
            print(f"Groq answer: {answer}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "="*50)
    print("Groq integration test complete!")

if __name__ == "__main__":
    test_groq_integration()