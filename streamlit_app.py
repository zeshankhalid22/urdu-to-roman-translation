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
        with open("static/urdu_vocab.json", "r", encoding="utf-8") as f:
            urdu_vocab_map = json.load(f)
        
        with open("static/roman_vocab.json", "r", encoding="utf-8") as f:
            roman_vocab_map = json.load(f)
            
        with open("static/urdu_merges.json", "r", encoding="utf-8") as f:
            urdu_merges = json.load(f)
        
        # Load model (adjust with your model loading code)
        model = torch.load("static/model.pt", map_location=torch.device('cpu'))
        
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
example_1 = """میری نوائے شوق سے شور حریم ذات میں غلغلہ ہائے الاماں بت کدۂ صفات میں حور و فرشتہ ہیں اسیر میرے تخیلات میں میری نگاہ سے خلل تیری تجلیات میں گرچہ ہے میری جستجو دیر و حرم کی نقشہ بند میری فغاں سے رستخیز کعبہ و سومنات میں گاہ مری نگاہ تیز چیر گئی دل وجود گاہ الجھ کے رہ گئی میرے توہمات میں تو نے یہ کیا غضب کیا مجھ کو بھی فاش کر دیا میں ہی تو ایک راز تھا سینۂ کائنات میں"""

example_2 = """سب رنگ میں اس گل کی مرے شان ہے موجود غافل تو ذرا دیکھ وہ ہر آن ہے موجود ہر تار کا دامن کے مرے کر کے تبرک سربستہ ہر اک خار بیابان ہے موجود عریانی تن ہے یہ بہ از خلعت شاہی ہم کو یہ ترے عشق میں سامان ہے موجود کس طرح لگاوے کوئی داماں کو ترے ہاتھ ہونے کو تو اب دست و گریبان ہے موجود لیتا ہی رہا رات ترے رخ کی بلائیں تو پوچھ لے یہ زلف پریشان ہے موجود تم چشم حقیقت سے اگر آپ کو دیکھو آئینۂ حق میں دل انسان ہے موجود کہتا ہے ظفرؔ ہیں یہ سخن آگے سبھوں کے جو کوئی یہاں صاحب عرفان ہے موجود"""

example_3 = """جب سے قریب ہو کے چلے زندگی سے ہم خود اپنے آئنے کو لگے اجنبی سے ہم کچھ دور چل کے راستے سب ایک سے لگے ملنے گئے کسی سے مل آئے کسی سے ہم اچھے برے کے فرق نے بستی اجاڑ دی مجبور ہو کے ملنے لگے ہر کسی سے ہم شائستہ محفلوں کی فضاؤں میں زہر تھا زندہ بچے ہیں ذہن کی آوارگی سے ہم اچھی بھلی تھی دنیا گزارے کے واسطے الجھے ہوئے ہیں اپنی ہی خود آگہی سے ہم جنگل میں دور تک کوئی دشمن نہ کوئی دوست مانوس ہو چلے ہیں مگر بمبئی سے ہم"""

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
