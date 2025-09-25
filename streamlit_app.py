import streamlit as st
import torch
import torch.nn as nn
import json
import re
import unicodedata
import random

# Set page title and configuration
st.set_page_config(
    page_title="Urdu-Roman Translator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #ffffff;
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1a237e;
        margin-bottom: 1rem;
    }
    
    /* Translation containers */
    .translation-box {
        background-color: #f5f7fa;
        border-radius: 5px;
        border-left: 3px solid #1a237e;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1a237e;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: normal;
    }
    
    .stButton button:hover {
        background-color: #3949ab;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        padding-top: 1rem;
    }
    
    /* Urdu text styling */
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
        font-size: 1.2rem;
        line-height: 2;
        text-align: right;
        direction: rtl;
    }
    
    /* Remove Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .reportview-container .main footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Model architecture (unchanged)
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

    def beam_search(self, src, beam_width=3, max_len=100):
        """Generate translations using beam search for better results."""
        batch_size = src.shape[0]
        sos_idx = 0  # Start token
        eos_idx = 2  # End token

        # Encode the source sequence
        encoder_outputs, hidden, cell = self.encoder(src)

        # Prepare decoder states
        hidden = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        # Initialize beam
        # Each beam contains: (cumulative_score, sequence, current_input, hidden, cell, token_history)
        beams = [(0, [sos_idx], torch.tensor([sos_idx], device=src.device), 
                  hidden, cell, set())]
        completed_beams = []

        # Beam search
        for _ in range(max_len):
            candidates = []

            for score, seq, inp, h, c, history in beams:
                # If sequence ends with EOS, add to completed
                if seq[-1] == eos_idx:
                    completed_beams.append((score, seq))
                    continue

                # Pass through decoder with attention
                output, new_h, new_c, _ = self.decoder(inp, h, c, encoder_outputs)

                # Get top-k probabilities and tokens
                log_probs, top_indices = output.log_softmax(1).topk(beam_width)

                # Create new candidates
                for i in range(beam_width):
                    token = top_indices[0, i].item()
                    log_prob = log_probs[0, i].item()

                    # Apply repetition penalty
                    rep_penalty = 1.0
                    if token in history:
                        rep_penalty = 0.7  # Reduce probability of repetition

                    # Create new sequence
                    new_seq = seq + [token]
                    new_score = score + log_prob * rep_penalty
                    new_input = torch.tensor([token], device=src.device)

                    # Add to history (keep recent history)
                    new_history = history.copy()
                    if len(new_history) > 5:
                        new_history.pop()
                    new_history.add(token)

                    candidates.append((new_score, new_seq, new_input, new_h, new_c, new_history))

                    # If token is EOS, also add to completed
                    if token == eos_idx:
                        completed_beams.append((new_score, new_seq))

            # Keep only top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            # Early stop if all beams have completed
            if len(beams) == 0:
                break

        # If no complete sequences found, use best incomplete one
        if not completed_beams and beams:
            completed_beams = [(beams[0][0], beams[0][1])]

        # Sort completed beams by score
        completed_beams = sorted(completed_beams, key=lambda x: x[0], reverse=True)

        # Return the best beam
        best_seq = completed_beams[0][1] if completed_beams else [sos_idx]

        # Create output tensor
        output = torch.zeros(1, max_len, dtype=torch.long, device=src.device)
        for i, token in enumerate(best_seq[:max_len]):
            output[0, i] = token

        return output

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

def translate_with_beam_search(text, model, urdu_merges, urdu_vocab_map, roman_vocab_map, beam_width=3, max_len=350):
    """Translate Urdu text to Roman Urdu using beam search."""
    try:
        # Tokenize
        _, indices = tokenize_urdu(text, urdu_merges, urdu_vocab_map)
        
        # Convert to tensor
        src_tensor = torch.tensor([indices], dtype=torch.long)
        
        # Translate with beam search
        with torch.no_grad():
            output = model.beam_search(src_tensor, beam_width=beam_width, max_len=max_len)
            
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
        return f"Translation error: {str(e)}"

# Main application header
st.title("Urdu to Roman Urdu Translator")
st.write("Neural machine translation using sequence-to-sequence model with attention mechanism")

# Define example texts
examples = {
    "Example 1": """دل نے وفا کے نام پر کار وفا نہیں کیا خود کو ہلاک کر لیا خود کو فدا نہیں کیا خیرہ سران شوق کا کوئی نہیں ہے جنبہ دار شہر میں اس گروہ نے کس کو خفا نہیں کیا جو بھی ہو تم پہ معترض اس کو یہی جواب دو آپ بہت شریف ہیں آپ نے کیا نہیں کیا نسبت علم ہے بہت حاکم وقت کو عزیز اس نے تو کار جہل بھی بے علما نہیں کیا جس کو بھی شیخ و شاہ نے حکم خدا دیا قرار ہم نے نہیں کیا وہ کام ہاں بہ خدا نہیں کیا""",
    
    "Example 2": """گر خامشی سے فائدہ اخفائے حال ہے خوش ہوں کہ میری بات سمجھنی محال ہے کس کو سناؤں حسرت اظہار کا گلہ دل فرد جمع و خرچ زباں ہائے لال ہے کس پردہ میں ہے آئنہ پرداز اے خدا رحمت کہ عذر خواہ لب بے سوال ہے ہے ہے خدا نخواستہ وہ اور دشمنی اے شوق منفعل یہ تجھے کیا خیال ہے مشکیں لباس کعبہ علی کے قدم سے جان ناف زمین ہے نہ کہ ناف غزال ہے وحشت پہ میری عرصۂ آفاق تنگ تھا دریا زمین کو عرق انفعال ہے ہستی کے مت فریب میں آ جائیو اسدؔ عالم تمام حلقۂ دام خیال ہے پہلو تہی نہ کر غم و اندوہ سے اسدؔ دل وقف درد کر کہ فقیروں کا مال ہے""",
    
    "Example 3": """بجا کہ آنکھ میں نیندوں کے سلسلے بھی نہیں شکست خواب کے اب مجھ میں حوصلے بھی نہیں نہیں نہیں یہ خبر دشمنوں نے دی ہوگی وہ آئے آ کے چلے بھی گئے ملے بھی نہیں یہ کون لوگ اندھیروں کی بات کرتے ہیں ابھی تو چاند تری یاد کے ڈھلے بھی نہیں ابھی سے میرے رفوگر کے ہاتھ تھکنے لگے ابھی تو چاک مرے زخم کے سلے بھی نہیں خفا اگرچہ ہمیشہ ہوئے مگر اب کے وہ برہمی ہے کہ ہم سے انہیں گلے بھی نہیں"""
}

# Create two columns for layout
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Examples")
    for name, text in examples.items():
        if st.button(name):
            st.session_state.urdu_text = text

    st.subheader("Translation Settings")
    beam_width = st.slider("Beam Width", min_value=1, max_value=5, value=3, 
                          help="Higher values may improve translation quality but take longer")
    
    max_len = st.slider("Max Length", min_value=100, max_value=500, value=350,
                       help="Maximum length of generated translation")


with col2:
    st.subheader("Input Text")
    
    # Check if there's a selected example, otherwise show input field
    if "urdu_text" not in st.session_state:
        st.session_state.urdu_text = ""
        
    # Text area for input
    urdu_text = st.text_area("Enter Urdu text:", 
                            value=st.session_state.urdu_text,
                            height=150)
    
    # Fixed beam width and max length
    beam_width = 3  # Fixed beam width
    max_len = 350  # Fixed max length
    
    if st.button("Translate"):
        if not urdu_text.strip():
            st.warning("Please enter some Urdu text to translate.")
        else:
            # Display Urdu text
            st.subheader("Original Urdu Text:")
            st.markdown(f'<div class="translation-box urdu-text">{urdu_text}</div>', unsafe_allow_html=True)
            
            # Load model
            with st.spinner("Loading model..."):
                model, urdu_vocab_map, roman_vocab_map, urdu_merges = load_model()
            
            if model:
                # Translate with beam search
                with st.spinner("Translating using beam search..."):
                    translation = translate_with_beam_search(
                        urdu_text, model, urdu_merges, urdu_vocab_map, roman_vocab_map,
                        beam_width=beam_width, max_len=max_len
                    )
                
                # Display translation
                st.subheader("Roman Urdu Translation:")
                st.markdown(f'<div class="translation-box">{translation}</div>', unsafe_allow_html=True)
            else:
                st.error("Error: Could not load the translation model.")

st.markdown("""
<div class="footer">
    <p>Urdu to Roman Urdu Neural Machine Translation | Developed by Zeeshan Khalid & Zahid Iqbal</p>
    <p>NLP Course Project | Instructor: Dr. Muhammad Usama</p>
</div>
""", unsafe_allow_html=True)

