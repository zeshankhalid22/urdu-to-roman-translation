import streamlit as st
import torch
import torch.nn as nn
import json
import re
import unicodedata

# Set page title and configuration
st.set_page_config(page_title="Urdu-Roman Translator", layout="centered")
st.title("Urdu to Roman Urdu Translation")

# Define your model architecture based on the parameters you provided
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        
        hidden_forward = hidden[0:self.num_layers]
        hidden_backward = hidden[self.num_layers:]
        cell_forward = cell[0:self.num_layers]
        cell_backward = cell[self.num_layers:]

        last_hidden_forward = hidden_forward[-1]
        last_hidden_backward = hidden_backward[-1]
        last_cell_forward = cell_forward[-1]
        last_cell_backward = cell_backward[-1]

        last_hidden = torch.cat([last_hidden_forward, last_hidden_backward], dim=1)
        last_cell = torch.cat([last_cell_forward, last_cell_backward], dim=1)

        dec_hidden = self.fc_hidden(last_hidden)
        dec_cell = self.fc_cell(last_cell)

        return outputs, dec_hidden, dec_cell

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return nn.functional.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embed_size, enc_hidden_size, dec_hidden_size, 
                 num_layers=2, dropout=0.1):
        super(AttentionDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = dec_hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        
        self.lstm = nn.LSTM(
            embed_size + (enc_hidden_size * 2),
            dec_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(dec_hidden_size + embed_size + (enc_hidden_size * 2), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # Handle input dimensions
        if input.dim() > 1:
            input = input.squeeze(-1)
        
        embedded = self.dropout(self.embedding(input))
        embedded = embedded.unsqueeze(1)
        
        top_hidden = hidden[-1]
        attn_weights = self.attention(top_hidden, encoder_outputs)
        
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, embedded, context), dim=1))
        
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # Not used for inference
        pass

    def translate(self, src, max_len=350):
        """Simple greedy decoding"""
        batch_size = src.shape[0]
        sos_token = 0  # Start token index
        eos_token = 2  # End token index
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, max_len, dtype=torch.long, device=src.device)
        outputs[:, 0] = sos_token
        
        # Encode the source
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Prepare decoder hidden state
        hidden = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        # Start with SOS token
        current_token = torch.tensor([sos_token], device=src.device)
        
        # Decode one token at a time
        for t in range(1, max_len):
            # Get output from decoder
            output, hidden, cell, _ = self.decoder(current_token, hidden, cell, encoder_outputs)
            
            # Get most likely next token
            current_token = output.argmax(1)
            
            # Save to output tensor
            outputs[:, t] = current_token
            
            # Stop if EOS token
            if current_token.item() == eos_token:
                break
        
        return outputs

# Helper function to load model
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
        
        # Model parameters
        input_size = len(urdu_vocab_map) + 1
        output_size = len(roman_vocab_map) + 1
        embed_size = 128
        hidden_size = 128
        enc_layers = 1
        dec_layers = 2
        dropout = 0.1
        device = torch.device('cpu')
        
        # Create model
        encoder = Encoder(input_size, embed_size, hidden_size, enc_layers, dropout)
        decoder = AttentionDecoder(output_size, embed_size, hidden_size, hidden_size, dec_layers, dropout)
        model = Seq2Seq(encoder, decoder, device)
        
        # Load state dict
        model_data = torch.load("static/model.pt", map_location=device)
        
        # Handle different saved formats
        if isinstance(model_data, dict) and 'model_state_dict' in model_data:
            model.load_state_dict(model_data['model_state_dict'])
        elif isinstance(model_data, dict):
            model.load_state_dict(model_data)
        else:
            # If model is already a model object, not a state dict
            model = model_data
            
        model.eval()
        
        return model, urdu_vocab_map, roman_vocab_map, urdu_merges
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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
            output = model.translate(src_tensor)
            
        # Convert output indices to text
        translation = []
        for idx in output[0, 1:].cpu().numpy():  # Skip <sos>
            if idx >= len(roman_vocab_map) or idx == 0:  # Skip padding and unknown
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
        import traceback
        return f"Translation error: {str(e)}\n{traceback.format_exc()}"

# Example ghazals
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
