import pickle
import re

import torch
from dateutil import parser
from transformers import DistilBertForSequenceClassification


class InferenceTool:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self._tokenizer = None
        self._label_encoder = None
        self._reverse_label_encoder = None
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return

        if self.model_path is None:
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(
                current_dir, "..", "model", "chatbot_model.pkl"
            )

        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please ensure the model file exists or train the model first."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model file at {self.model_path}: {str(e)}"
            )

        # Check if model_data is a dictionary (expected format)
        if not isinstance(model_data, dict):
            raise ValueError(
                f"Expected model_data to be a dictionary, but got {type(model_data)}. "
                f"The pickle file appears to contain a {type(model_data).__name__} object. "
                "Please ensure you're using the correct model file that was saved with the expected format: "
                "{'tokenizer': ..., 'label_encoder': ..., 'reverse_label_encoder': ..., 'model_state_dict': ...}. "
                "If you have a sklearn Pipeline model, you may need to regenerate the model file with the correct format."
            )

        # Validate required keys
        required_keys = ["tokenizer", "label_encoder", "reverse_label_encoder", "model_state_dict"]
        missing_keys = [key for key in required_keys if key not in model_data]
        if missing_keys:
            raise ValueError(
                f"Model data is missing required keys: {missing_keys}. "
                f"Available keys: {list(model_data.keys())}. "
                "Please regenerate the model file with all required components."
            )

        self._tokenizer = model_data["tokenizer"]
        self._label_encoder = model_data["label_encoder"]
        self._reverse_label_encoder = model_data["reverse_label_encoder"]

        num_labels = len(self._label_encoder)
        self._model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        self._model.load_state_dict(model_data["model_state_dict"])
        self._model.eval()
        self._initialized = True

    @property
    def tokenizer(self):
        self._ensure_initialized()
        return self._tokenizer

    @property
    def label_encoder(self):
        self._ensure_initialized()
        return self._label_encoder

    @property
    def reverse_label_encoder(self):
        self._ensure_initialized()
        return self._reverse_label_encoder

    @property
    def model(self):
        self._ensure_initialized()
        return self._model

    def predict_intent(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_label].item()

        intent = self.reverse_label_encoder[predicted_label]
        return intent, confidence

    def extract_datetime(self, text):
        try:
            parsed_date = parser.parse(text, fuzzy=True)
            standardized = parsed_date.strftime("%Y-%m-%d %H:%M")
            return standardized
        except ValueError:
            return None

    def predict_and_respond(self, text):
        intent, confidence = self.predict_intent(text)

        responses = {
            "greeting": "Hello! How can I help with your booking?",
            "reschedule_booking": "Sure, let's reschedule. Provide the new date and time.",
            "cancel_booking": "Got it. Confirm if you want to cancel.",
            "pricing_inquiry": "Let me check the prices.",
            "book_service": "I'd be happy to book. What type and when?",
            "booking_status": "Please provide your booking reference.",
            "thanks": "You're welcome!",
            "confirm": "Confirmed!",
            "deny": "No problem.",
            "provide_datetime": "Noted the time.",
        }

        response = responses.get(
            intent, "I'm sorry, I didn't understand that."
        )
        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
        }