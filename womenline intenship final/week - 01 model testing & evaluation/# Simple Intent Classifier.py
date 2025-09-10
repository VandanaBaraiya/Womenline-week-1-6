# Simple Intent Classifier

def classify_intent(question):
    question = question.lower()

    # Define keywords per intent
    intent_keywords = {
        "nutrition": ["eat", "food", "diet", "banana", "hydration", "fruits"],
        "medicine": ["paracetamol", "antibiotic", "tablet", "medicine", "safe"],
        "emergency": ["fever", "dengue", "food poisoning", "emergency", "pain", "vomit"],
        "general_health": ["blood pressure", "baby", "signs", "symptoms"]
    }

    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword in question:
                return intent

    return "unknown"

# Test Example
questions = [
    "What to eat when hemoglobin is low?",
    "Is paracetamol safe during pregnancy?",
    "What to do in case of food poisoning?",
    "Can diabetics eat bananas?",
    "What are early signs of dengue?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"→ Intent: {classify_intent(q)}\n")