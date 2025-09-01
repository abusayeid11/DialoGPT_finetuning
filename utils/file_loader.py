import json
import os
from glob import glob
from difflib import get_close_matches


SUBTOPICS = {
    "pros": ["advantages", "benefits", "upsides"],
    "cons": ["disadvantages", "drawbacks"],
    "resources": ["books", "courses", "learn", "study"],
    "steps": ["roadmap", "how to start", "path"]
}

def detect_subtopic(user_input):
    user_input = user_input.lower()
    for subtopic, keywords in SUBTOPICS.items():
        if any(keyword in user_input for keyword in keywords):
            return subtopic
    return None

# utils/file_loader.py
def load_knowledge_base(dir_path="knowledge_base"):
    knowledge = {}
    for file_path in glob(os.path.join(dir_path, "*.json")):
        topic = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Add intent keywords to each topic
            data["_intents"] = [
                topic.replace("_", " "),  # "software_eng" -> "software eng"
                *data.get("aliases", [])  # Custom aliases (add to JSON files)
            ]
            knowledge[topic] = data
    return knowledge

def get_best_match(user_input, knowledge):
    user_input = user_input.lower()
    for topic, data in knowledge.items():
        # Check if any intent keyword matches
        if any(keyword in user_input for keyword in data["_intents"]):
            return topic
    return None