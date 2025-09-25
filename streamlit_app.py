import streamlit as st
import torch
import json
import re
import unicodedata

# Set page title and configuration
st.set_page_config(page_title="Urdu-Roman Translator", layout="centered")
st.title("Urdu to Roman Urdu Translation")

# Load model and vocabulary (adjust paths as needed)
@st.cache_resource
def load_model():
    try:
        # Load vocabularies
        with open("urdu_vocab_improved.json", "r", encoding="utf-8") as f:
            urdu_vocab_map = json.load(f)
        
        with open("roman_vocab.json", "r", encoding="utf-8") as f:
            roman_vocab_map = json.load(f)
            
        with open("urdu_merges_improved.json", "r", encoding="utf-8") as f:
            urdu_merges = json.load(f)
        
        # Load model (adjust with your model loading code)
        model = torch.load("best_model.pt", map_location=torch.device('cpu'))
        
        return model, urdu_vocab_map, roman_vocab_map, urdu_merges
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Helper functions
def merge_pair(tokens, pair, new_token):
    """Merge a specific pair in a sequence."""
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
            result.append(new_token)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def tokenize_urdu(text, urdu_merges, urdu_vocab_map):
    """Tokenize Urdu text with word boundary markers."""
    # Clean text
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0900-\u097F\s\n]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize with word boundaries
    tokens = []
    for word in text.split():
        tokens.append('▁')  # Word boundary marker
        for char in word:
            tokens.append(char)
    
    tokens.append('<eos>')
    
    # Apply merges
    for pair, new_token in urdu_merges:
        tokens = merge_pair(tokens, tuple(pair), new_token)
    
    # Convert to indices
    indices = [urdu_vocab_map.get(token, len(urdu_vocab_map)) for token in tokens]
    
    return tokens, indices

def translate_text(text, model, urdu_merges, urdu_vocab_map, roman_vocab_map):
    """Translate Urdu text to Roman Urdu."""
    try:
        # Tokenize
        _, indices = tokenize_urdu(text, urdu_merges, urdu_vocab_map)
        
        # Convert to tensor
        src_tensor = torch.tensor([indices], dtype=torch.long)
        
        # Translate
        with torch.no_grad():
            # Assuming your model has beam_search or translate method
            output = model.beam_search(src_tensor, beam_width=3, max_len=350)
            
        # Convert output indices to text
        translation = []
        for idx in output[0, 1:].cpu().numpy():  # Skip <sos>
            if idx >= len(roman_vocab_map):  # Skip padding
                continue
            token = next((k for k, v in roman_vocab_map.items() if v == idx), None)
            if token == '<eos>':
                break
            if token:
                translation.append(token)
        
        # Join tokens
        roman_text = ''.join(translation).replace('_', ' ')
        return roman_text
    except Exception as e:
        return f"Translation error: {str(e)}"

# Example ghazals (replace with your examples)
example_1 = """نہ کرتے اضطرابِ وحشت اب تو ہم کہ مر گئے
مزے سے ہائے دکھ میں کٹ گئی عمر اس طرح گئے"""

example_2 = """اپنی اداؤں پہ دنیا فدا کر دی 
فصل بہاراں کو خزاں میں بدل دیا"""

example_3 = """محبت بھرا دل بڑی چیز ہے
وفا بھرا انسان لاجواب ہے"""

# Create columns for buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example 1"):
        st.session_state.selected_text = example_1

with col2:
    if st.button("Example 2"):
        st.session_state.selected_text = example_2

with col3:
    if st.button("Example 3"):
        st.session_state.selected_text = example_3

# Display selected text and translation
if "selected_text" in st.session_state:
    # Display selected Urdu text
    st.subheader("Urdu Text:")
    st.write(st.session_state.selected_text)
    
    # Load model if not already loaded
    if "model_loaded" not in st.session_state:
        with st.spinner("Loading model..."):
            model, urdu_vocab_map, roman_vocab_map, urdu_merges = load_model()
            if model:
                st.session_state.model = model
                st.session_state.urdu_vocab_map = urdu_vocab_map
                st.session_state.roman_vocab_map = roman_vocab_map
                st.session_state.urdu_merges = urdu_merges
                st.session_state.model_loaded = True
    
    # Translate and display
    if "model_loaded" in st.session_state:
        with st.spinner("Translating..."):
            translation = translate_text(
                st.session_state.selected_text,
                st.session_state.model,
                st.session_state.urdu_merges,
                st.session_state.urdu_vocab_map,
                st.session_state.roman_vocab_map
            )
        
        # Display translation
        st.subheader("Roman Urdu Translation:")
        st.write(translation)
else:
    st.write("Click on any example button to see the translation.")
