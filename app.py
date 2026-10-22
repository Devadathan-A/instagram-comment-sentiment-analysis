# =============================================================================
# app.py — Instagram Comment Sentiment Analysis
# Upload your CSV file directly inside the webpage — no command line needed
#
# HOW TO RUN:
#   python app.py
#   Then open your browser at: http://127.0.0.1:5000
#
# INSTALL ONCE:
#   pip install nltk pandas matplotlib flask textblob emoji
# =============================================================================

import io, re, base64, threading, webbrowser
import nltk, pandas as pd, matplotlib, matplotlib.pyplot as plt
from collections import Counter
from flask import Flask, request, jsonify, render_template_string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
try:
    import emoji
    EMOJI_OK = True
except ImportError:
    EMOJI_OK = False

matplotlib.use('Agg')

print("Setting up NLTK resources...")
for r in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon', 'punkt_tab', 'words']:
    nltk.download(r, quiet=True)
print("Ready.\n")

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

CUSTOM_IGNORE = {
    # Brand names — never carry sentiment or aspect meaning
    'nike','adidas','puma','zara','amazon','flipkart','myntra',
    'samsung','apple','sony','lg','boat','jbl','bose','oneplus',
    'realme','oppo','vivo','xiaomi','redmi','asus','dell','hp',
    'lenovo','logitech','philips','bosch','ikea',

    # Generic transaction/filler words with no sentiment signal
    # NOTE: 'price','delivery','service','design','material','color','colour',
    # 'shipping','box','cost','paid','size','look','looks','like' are
    # intentionally KEPT OUT — they are aspect trigger keywords used in
    # ASPECTS{} and show up in keyword clouds, so removing them would
    # silently break aspect detection and weaken keyword reporting.
    'product','item','order','package',
    'bought','buy','purchase','money',
    'received','arrived','came','come','get','got','use',
    'using','used','day','week','month','time','one','two',
    'first','second',

    # Pure filler — no sentiment or aspect value whatsoever
    'also','even','still','just','really','very','much',
    'lot','bit','thing','things','something','nothing','everything',
    'would','could','said','say','seem',
}
stop_words = stop_words | CUSTOM_IGNORE
sia = SentimentIntensityAnalyzer()

from nltk.corpus import words as nltk_words_corpus

# Full English vocabulary: 236k+ real English words + stopwords
# Used as the primary gate in is_english()
ENGLISH_VOCAB = set(w.lower() for w in nltk_words_corpus.words()) | set(stopwords.words('english'))

# Kept for legacy reference (no longer used as the sole check)
ENGLISH_WORDS = set(stopwords.words('english'))

# Hinglish detection — Hindi words commonly written in Roman/English script
HINDI_MARKERS = {
    'karo','kara','kyu','phir','mila','nahi','nhi','baat','mne',
    'kharid','galat','abhi','yaar','wala','ghar','leke','cahea',
    'bikega','kaam','chahate','saab','jaye','gand','waah','kaa',
    'laate','bawasir','bna','abey','gareebo','dedi','toh','mara',
    'carna','chalta','banna','chata','boleh','lah','hitam','bhen',
    'loriii','chaiye','acchcha','lagta','lekin','zarurat','kudunga',
    'bkwas','bekar','bakwas','bohat','ghatiya','bilkul','ekadam',
    'nahin','camre','malum','biker','sala','beker','mlki','hora',
    'nho','kuch','baccho','mereko','agar','sakte','raha','rahi',
    'maine','liya','mein','bahut','karke','koi','karne','lounch',
    'doo','lengend','pesa','dono','hota','tha','ki','jaldi','bhi',
    'hoga','jara','batching','chaiye','zarurat','thi','mera','kya',
}

# Common non-English function words that reliably signal other languages.
# These are words that would NOT appear naturally in an English sentence.
NON_ENGLISH_MARKERS = {
    # Spanish
    'que','pero','para','con','una','del','los','las','por','como',
    'mas','sin','hay','este','esta','esto','estos','estas','muy',
    'tambien','seria','genial','hacerlo','estaría','probaran',
    'barras','invertidas','mexico','seria',
    # French
    'les','des','une','dans','est','pas','sur','avec','pour','qui',
    'que','sont','tout','mais','bien','tres','plus',
    # Portuguese
    'que','nao','para','com','uma','dos','nas','pelos','pelas',
    # Malay / Indonesian
    'yang','tidak','dengan','untuk','dari','akan','dalam','adalah',
    # General romanized Hindi/Urdu not in HINDI_MARKERS yet
    'lagao','kitna','rahega','niche','dedo','merko',
}


# Aspect keyword groups
ASPECTS = {
    'Quality':  ['quality','build','durable','durability','material','finish','sturdy','cheap','flimsy','premium','solid','fragile','broke','broken','defect','defective'],
    'Price':    ['price','cost','worth','value','expensive','cheap','affordable','overpriced','budget','deal','money','paid','pricey'],
    'Delivery': ['delivery','shipping','arrived','fast','slow','delayed','late','courier','dispatch','packaging','packed','box','damage','damaged'],
    'Service':  ['service','support','response','refund','return','helpful','rude','staff','team','customer','care','assist'],
    'Design':   ['design','look','style','color','colour','beautiful','ugly','attractive','sleek','appearance','aesthetic','orange','blossom','black','white','silver','gold','titanium','stellar','gradient','matte','glossy'],
}

# Sarcasm patterns
SARCASM_PATTERNS = [
    r'\boh wow\b', r'\boh great\b', r'\bsure\b.{0,15}\!',
    r'\bso helpful\b', r'\bthanks a lot\b.{0,10}[!?]',
    r'\bwhat a (deal|bargain|joke|surprise)\b',
    r'\byeah right\b', r'\bsuper (great|amazing|wonderful|helpful)\b.{0,5}[!]',
]
SARCASM_RE = [re.compile(p, re.IGNORECASE) for p in SARCASM_PATTERNS]

SARCASM_EMOJI = {'🙄','😒','😑','🤦','🤷','😤','💀','☠️','🥴'}

def is_english(text):
    """
    Strict English detection. Returns True only if the comment is
    confidently English. Rejects non-Latin scripts, Hinglish, Spanish,
    French, and other Roman-script languages.

    Strategy:
      1. Reject non-Latin scripts (Arabic, Devanagari, CJK …)
      2. Reject if any Hinglish or known non-English marker word is present
      3. Extract all alphabetic tokens ≥ 3 chars, ignoring brand names/numbers
      4. Require that at least MIN_RATIO of those tokens exist in the
         236k-word NLTK English vocabulary.
         — Short comments (< MIN_TOKENS meaningful words) get a looser ratio
           so genuine short English comments ("great product!") still pass.
    """
    text = str(text).strip()
    if not text:
        return False

    # ── Step 1: reject non-Latin scripts ──────────────────────────────────
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if ord(c) < 128]
    if len(ascii_letters) / len(letters) < 0.8:
        return False

    # ── Step 2: reject Hinglish and other known non-English markers ────────
    words = re.findall(r'[a-z]+', text.lower())
    if any(w in HINDI_MARKERS for w in words):
        return False
    if any(w in NON_ENGLISH_MARKERS for w in words):
        return False

    # ── Step 3: vocabulary ratio check ────────────────────────────────────
    meaningful = [
        w for w in words
        if len(w) >= 3 and w not in BRAND_SKIP
    ]

    if not meaningful:
        # Only brand names / very short tokens — not enough signal
        return False

    in_vocab   = sum(1 for w in meaningful if w in ENGLISH_VOCAB)
    ratio      = in_vocab / len(meaningful)

    # Threshold: stricter for longer comments, a bit looser for very short ones
    if len(meaningful) >= 6:
        threshold = 0.40   # 40% of tokens must be real English words
    elif len(meaningful) >= 3:
        threshold = 0.30   # 30% for medium-short comments
    else:
        # 1-2 meaningful words: require at least one to be English
        threshold = 0.50

    return ratio >= threshold

# Module-level brand/product name list used by both is_english() and WORD_SKIP
BRAND_SKIP = {
    'oppo','bajaj','yamaha','honda','suzuki','kawasaki','tvs','ktm',
    'samsung','apple','sony','oneplus','realme','vivo','xiaomi','redmi',
    'asus','dell','hp','lenovo','nike','adidas','puma','ikea','bosch',
    'ns160','ns200','n160','cg','dtsi',
}

# Compiled once at startup — not rebuilt on every request
FILTER_RE = [re.compile(p, re.IGNORECASE) for p in [
    r'^\s*(please|pls|plz)\b',
    r'\bplease\s+(gift|send|give|make|launch|release|available|bring|add|provide|collab|feature)\b',
    r'^\s*(price\s*\??|cost\s*\??|how\s+much|rate\s*\??)\s*$',
    r'^\s*(can\s+you|could\s+you|will\s+you|when\s+will|when\s+is|any\s+chance)\b',
    r'^\s*(available|availability|launch|release)\b',
    r'^\s*(dear\s+\w+|hi\s*,|hello\s*,)',
    r'\bwe\s+(value|appreciate|understand|regret)\b',
    r'\bwe\s+request\s+you\b',
    r'\bplease\s+(dm|direct\s+message|contact\s+us|reach\s+out|connect\s+with\s+us)\b',
    r'\bshare\s+your\s+(detailed\s+)?feedback\b',
]]

def is_irrelevant(text):
    t = text.strip()
    if re.match(r'^\s*@\w+', t): return True
    if re.match(r'^\w[\w\s]{0,15}@\w+', t): return True
    if any(p.search(t) for p in FILTER_RE): return True
    # FIX: filter out very short comments (< 4 words) — too little signal
    if len(re.findall(r'\b\w+\b', t)) < 4: return True
    return False

def handle_emojis(text):
    """Convert emojis to descriptive text for better sentiment capture."""
    if EMOJI_OK:
        return emoji.demojize(text, delimiters=(' ', ' '))
    return text

def detect_sarcasm(text):
    """Heuristic sarcasm detection with reduced false positives."""
    for pattern in SARCASM_RE:
        if pattern.search(text):
            return True
    for em in SARCASM_EMOJI:
        if em in text:
            return True
    # FIX: positive+negative co-occurrence needs proximity — words must appear
    # within 6 tokens of each other to count (avoids "not bad... but amazing value")
    tokens = re.findall(r'\b\w+\b', text.lower())
    POS_W = {'great','amazing','wonderful','excellent','fantastic','awesome'}
    NEG_W = {'not','never','worst','bad','terrible','horrible','awful','useless'}
    for i, tok in enumerate(tokens):
        if tok in POS_W:
            window = tokens[max(0, i-6):i+7]
            if any(w in NEG_W for w in window):
                return True
    return False

def preprocess(text):
    text   = str(text).lower()
    text   = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text   = re.sub(r'[^a-z\s]', '', text)
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def analyse_aspects(comments_series, labels_series):
    """Detect aspect-level sentiment and extract meaningful specific words per aspect."""
    WORD_SKIP = {
        'the','and','for','that','this','with','are','was','its','from','have',
        'they','will','been','not','but','all','can','had','get','got','one','two',
        'also','just','very','much','more','too','like','please','want','need',
        'even','still','know','only','than','then','some','make','take','come',
        'what','when','where','would','could','should','about','their','there',
        'your','our','who','how','into','back','most','such','each','both','you',
        'has','her','him','his','now','new','buy','use','wow','great','nice',
        'love','really','phone','opinion','contact','india','global','instagram',
        'facebook','twitter','bought','purchase','product','item','order','good','bad',
    } | BRAND_SKIP

    GENERIC_ASPECT_WORDS = {
        'quality','build','durable','durability','material','finish','sturdy','flimsy',
        'premium','solid','fragile','defect','defective','price','cost','worth','value',
        'expensive','affordable','overpriced','budget','deal','paid','pricey','delivery',
        'shipping','arrived','fast','slow','delayed','late','courier','dispatch',
        'packaging','packed','damage','damaged','service','support','response','refund',
        'return','helpful','rude','staff','team','customer','care','assist',
        'design','style','beautiful','ugly','attractive','sleek','appearance','aesthetic',
        'look','looks','color','colour','matte','glossy','gradient',
    }

    def extract_specifics(comment_list, n=3):
        """Extract the most meaningful descriptive words from a set of comments."""
        all_words = []
        for c in comment_list:
            all_words += re.findall(r"[a-z]{4,}", c.lower())
        freq = Counter(
            w for w in all_words
            if w not in WORD_SKIP and w not in GENERIC_ASPECT_WORDS
        )
        return [w for w, cnt in freq.most_common(n) if cnt >= 2]

    results = {}
    for aspect, keywords in ASPECTS.items():
        pos = neg = neu = 0
        pos_comments, neg_comments = [], []
        kw_patterns = [re.compile(r'\b' + re.escape(kw) + r'\b') for kw in keywords]

        for comment, label in zip(comments_series, labels_series):
            cl = comment.lower()
            if any(p.search(cl) for p in kw_patterns):
                if label == 'Positive':
                    pos += 1; pos_comments.append(comment)
                elif label == 'Negative':
                    neg += 1; neg_comments.append(comment)
                else:
                    neu += 1

        total = pos + neg + neu
        if total > 0:
            pos_specifics = extract_specifics(pos_comments)
            neg_specifics = extract_specifics(neg_comments)
            results[aspect] = {
                'pos': pos, 'neg': neg, 'neu': neu, 'total': total,
                'pos_specifics': pos_specifics,
                'neg_specifics': neg_specifics,
                'specifics': pos_specifics or neg_specifics,  # backward compat
            }
    return results



def generate_nl_summary(total, pos_count, neg_count, neu_count,
                        pos_pct, neg_pct, neu_pct, pos_words, neg_words,
                        all_comments=None, all_labels=None, aspects=None):

    top_pos = [w for w, _ in pos_words[:5]]
    top_neg = [w for w, _ in neg_words[:5]]

    # ── Overall tone ──────────────────────────────────────────────────────
    if pos_pct >= 70:
        tone = "overwhelmingly positive"
        tone_detail = (
            f"Customers are highly satisfied — {pos_pct}% of all comments expressed positive sentiment. "
            f"This indicates strong brand health and a product that is clearly meeting expectations."
        )
    elif pos_pct >= 55:
        tone = "mostly positive"
        tone_detail = (
            f"The majority of customers ({pos_pct}%) are happy with their experience, "
            f"though {neg_pct}% expressed concerns that are worth investigating."
        )
    elif neg_pct >= 70:
        tone = "overwhelmingly negative"
        tone_detail = (
            f"A significant {neg_pct}% of comments are negative — this is a serious signal. "
            f"Customers are clearly dissatisfied and the brand needs to act urgently."
        )
    elif neg_pct >= 55:
        tone = "mostly negative"
        tone_detail = (
            f"More customers are unhappy ({neg_pct}%) than satisfied ({pos_pct}%). "
            f"Key pain points need to be identified and addressed before reputation suffers further."
        )
    elif neu_pct >= 40:
        tone = "largely neutral"
        tone_detail = (
            f"{neu_pct}% of comments are neutral, suggesting customers have low emotional engagement "
            f"with the product — neither delighted nor frustrated."
        )
    else:
        tone = "mixed"
        tone_detail = (
            f"Customer opinions are divided: {pos_pct}% positive, {neg_pct}% negative, "
            f"and {neu_pct}% neutral. The experience is inconsistent across customers."
        )

    summary_lines = [
        f"Out of <strong>{total}</strong> English comments analysed, the overall sentiment is "
        f"<strong>{tone}</strong>. {tone_detail}"
    ]

    # ── Aspect-level insights (the most meaningful section) ───────────────
    if aspects:
        aspect_insights = []
        for aspect, data in aspects.items():
            if data['pos'] + data['neg'] >= 2:
                aspect_insights.append((aspect, data))
        aspect_insights.sort(key=lambda x: x[1]['pos'] + x[1]['neg'], reverse=True)

        praised    = [(a, d) for a, d in aspect_insights if d['pos'] > d['neg']]
        complained = [(a, d) for a, d in aspect_insights if d['neg'] > d['pos']]
        divided    = [(a, d) for a, d in aspect_insights if d['neg'] == d['pos'] and d['pos'] > 0]

        # ── Praised aspects ──
        for aspect, data in praised[:2]:
            p, n = data['pos'], data['neg']
            pos_spec = data.get('pos_specifics', [])
            ratio_str = f"{p} positive vs {n} negative mention{'s' if n != 1 else ''}"

            if aspect == 'Quality':
                spec_note = f", especially praising its <em>{', '.join(pos_spec)}</em>" if pos_spec else ""
                summary_lines.append(
                    f"<strong>Build quality</strong> is a standout strength{spec_note} — {ratio_str}. "
                    f"Customers feel the product is well-made and reliable."
                )
            elif aspect == 'Price':
                spec_note = f", highlighting <em>{', '.join(pos_spec)}</em>" if pos_spec else ""
                summary_lines.append(
                    f"<strong>Pricing</strong> is viewed positively{spec_note} ({ratio_str}). "
                    f"Customers consider the product good value for money."
                )
            elif aspect == 'Delivery':
                spec_note = f", noting <em>{', '.join(pos_spec)}</em>" if pos_spec else ""
                summary_lines.append(
                    f"<strong>Delivery experience</strong> is praised{spec_note} ({ratio_str}). "
                    f"Customers are happy with how their orders arrived."
                )
            elif aspect == 'Service':
                spec_note = f", particularly around <em>{', '.join(pos_spec)}</em>" if pos_spec else ""
                summary_lines.append(
                    f"<strong>Customer service</strong> is a positive differentiator{spec_note} ({ratio_str}). "
                    f"Customers feel well-supported and their issues are being resolved."
                )
            elif aspect == 'Design':
                spec_note = f", with customers specifically mentioning <em>{', '.join(pos_spec)}</em>" if pos_spec else ""
                summary_lines.append(
                    f"<strong>Design and appearance</strong> are well-received{spec_note} ({ratio_str}). "
                    f"The aesthetic and look of the product are clear selling points."
                )

        # ── Complained aspects ──
        for aspect, data in complained[:2]:
            p, n = data['pos'], data['neg']
            neg_spec = data.get('neg_specifics', [])
            ratio_str = f"{n} negative vs {p} positive mention{'s' if p != 1 else ''}"

            if aspect == 'Quality':
                spec_note = f" — customers specifically mention <em>{', '.join(neg_spec)}</em>" if neg_spec else ""
                summary_lines.append(
                    f"<strong>Build quality</strong> is the top concern{spec_note} ({ratio_str}). "
                    f"Reports of defects, damage, or poor materials are a red flag that needs investigation."
                )
            elif aspect == 'Price':
                spec_note = f", with recurring mentions of <em>{', '.join(neg_spec)}</em>" if neg_spec else ""
                summary_lines.append(
                    f"<strong>Pricing</strong> is drawing criticism{spec_note} ({ratio_str}). "
                    f"Customers feel the product is overpriced relative to what they received."
                )
            elif aspect == 'Delivery':
                spec_note = f" — issues around <em>{', '.join(neg_spec)}</em> are mentioned repeatedly" if neg_spec else ""
                summary_lines.append(
                    f"<strong>Delivery</strong> is a recurring pain point{spec_note} ({ratio_str}). "
                    f"Late arrivals, damaged packaging, or slow dispatch are hurting the post-purchase experience."
                )
            elif aspect == 'Service':
                spec_note = f", particularly around <em>{', '.join(neg_spec)}</em>" if neg_spec else ""
                summary_lines.append(
                    f"<strong>Customer service</strong> is a significant complaint area{spec_note} ({ratio_str}). "
                    f"Customers feel unheard or unsupported — this directly impacts brand loyalty."
                )
            elif aspect == 'Design':
                spec_note = f", with customers noting <em>{', '.join(neg_spec)}</em>" if neg_spec else ""
                summary_lines.append(
                    f"<strong>Design</strong> is receiving negative feedback{spec_note} ({ratio_str}). "
                    f"Aesthetic concerns or usability issues with the product's appearance are being raised."
                )

        # ── Divided aspects ──
        for aspect, data in divided[:1]:
            p = data['pos']
            summary_lines.append(
                f"<strong>{aspect}</strong> divides opinion equally ({p} positive and {p} negative mentions), "
                f"suggesting the experience here varies significantly from customer to customer — "
                f"possibly depending on location, timing, or product variant."
            )

    # ── Keyword insight ───────────────────────────────────────────────────
    if top_pos and pos_count >= 5:
        summary_lines.append(
            f"The most frequent words in positive comments are "
            f"<em>{', '.join(top_pos[:4])}</em> — these themes represent what the brand does well "
            f"and should be highlighted in marketing and product descriptions."
        )
    if top_neg and neg_count >= 5:
        summary_lines.append(
            f"Negative comments cluster around words like <em>{', '.join(top_neg[:4])}</em>. "
            f"These are specific signals of what customers are dissatisfied with and should be "
            f"prioritised in the product or support team's improvement roadmap."
        )

    # ── Actionable suggestions ────────────────────────────────────────────
    suggestions = []

    if pos_pct >= 55 and top_pos:
        suggestions.append(
            f"Amplify what's working: terms like <em>{', '.join(top_pos[:3])}</em> appear repeatedly "
            f"in satisfied customer feedback. Use these as anchors in your ad copy, product pages, and testimonials."
        )
    if neg_pct >= 20 and top_neg:
        suggestions.append(
            f"Create a targeted fix list from negative feedback. The words "
            f"<em>{', '.join(top_neg[:3])}</em> point to specific, recurring issues — "
            f"not vague dissatisfaction. These can be actioned by product, logistics, or support teams directly."
        )
    if neu_pct >= 25:
        suggestions.append(
            f"{neu_count} customers left neutral feedback — they are not unhappy, but they are not advocates either. "
            f"A targeted follow-up (discount, personalised message, or proactive check-in) "
            f"could convert a meaningful portion of them into positive reviewers."
        )
    if neg_pct >= 40:
        suggestions.append(
            f"With {neg_pct}% negative sentiment, consider a public response strategy: "
            f"acknowledge the concerns openly, state what is being fixed, and set a timeline. "
            f"Transparent communication during a trust deficit consistently outperforms silence."
        )
    if pos_pct >= 65:
        suggestions.append(
            f"Your satisfaction rate of {pos_pct}% is strong enough to run a review campaign. "
            f"Reach out to recent buyers and prompt them to leave reviews on Google, Trustpilot, or Amazon — "
            f"happy customers rarely volunteer reviews without a nudge."
        )
    if aspects:
        complained = [(a, d) for a, d in aspects.items() if d['neg'] > d['pos'] and d['neg'] >= 3]
        if complained:
            worst = max(complained, key=lambda x: x[1]['neg'])
            suggestions.append(
                f"<strong>{worst[0]}</strong> is the highest-volume complaint area with "
                f"{worst[1]['neg']} negative mentions. Prioritise this above other improvements — "
                f"fixing a concentrated pain point has more impact than spreading effort across many small ones."
            )

    return {'summary_paras': summary_lines, 'suggestions': suggestions, 'tone': tone}


def analyse(comments):
    """Full NLP pipeline."""
    # Handle emojis before preprocessing AND before scoring
    # (VADER understands demojized text like "smiling_face_with_hearts")
    emoji_handled = [handle_emojis(c) for c in comments]
    cleaned       = [preprocess(c) for c in emoji_handled]

    # FIX: score the emoji-handled text, not the raw comment
    vader_scores    = [round(sia.polarity_scores(c)['compound'], 4) for c in emoji_handled]
    textblob_scores = [round(TextBlob(c).sentiment.polarity, 4)     for c in emoji_handled]
    # FIX: weighted average — VADER is stronger for short social-media text
    scores          = [round(0.65 * v + 0.35 * t, 4) for v, t in zip(vader_scores, textblob_scores)]

    def classify(s):
        if s >= 0.05:  return 'Positive'
        if s <= -0.05: return 'Negative'
        return 'Neutral'

    # FIX: 3-tier confidence (Low / Medium / High) instead of binary
    def confidence(s):
        a = abs(s)
        if a >= 0.40: return 'High'
        if a >= 0.15: return 'Medium'
        return 'Low'

    labels      = [classify(s)   for s in scores]
    confidences = [confidence(s) for s in scores]
    sarcasms    = [detect_sarcasm(c) for c in comments]

    # FIX: when sarcasm is detected, flip Positive → Negative and invert score
    # so the label actually reflects the true sentiment
    for i, sarc in enumerate(sarcasms):
        if sarc and labels[i] == 'Positive':
            labels[i] = 'Negative'
            scores[i] = round(-abs(scores[i]), 4)

    df = pd.DataFrame({
        'Comment':         comments,
        'Cleaned_Comment': cleaned,
        'VADER_Score':     vader_scores,
        'TextBlob_Score':  textblob_scores,
        'Sentiment_Score': scores,
        'Sentiment_Label': labels,
        'Confidence':      confidences,
        'Sarcasm':         sarcasms,
    })

    pos_count = int((df['Sentiment_Label'] == 'Positive').sum())
    neg_count = int((df['Sentiment_Label'] == 'Negative').sum())
    neu_count = int((df['Sentiment_Label'] == 'Neutral').sum())
    sarc_count= int(df['Sarcasm'].sum())
    total     = len(df)
    pos_pct   = round(pos_count / total * 100, 1)
    neg_pct   = round(neg_count / total * 100, 1)
    neu_pct   = round(neu_count / total * 100, 1)

    def top_words(label, n=12):
        words = ' '.join(df[df['Sentiment_Label'] == label]['Cleaned_Comment']).split()
        return Counter(words).most_common(n)

    pos_words = top_words('Positive')
    neg_words = top_words('Negative')

    # Aspect analysis — run before summary so aspects can be used in report
    aspects = analyse_aspects(df['Comment'], df['Sentiment_Label'])

    nl = generate_nl_summary(total, pos_count, neg_count, neu_count,
                             pos_pct, neg_pct, neu_pct, pos_words, neg_words,
                             all_comments=df['Comment'].tolist(),
                             all_labels=df['Sentiment_Label'].tolist(),
                             aspects=aspects)

    # Top positive & negative comments
    pos_df = df[df['Sentiment_Label'] == 'Positive']
    neg_df = df[df['Sentiment_Label'] == 'Negative']
    top_pos_comment = pos_df.loc[pos_df['Sentiment_Score'].idxmax(), 'Comment'] if len(pos_df) > 0 else ''
    top_neg_comment = neg_df.loc[neg_df['Sentiment_Score'].idxmin(), 'Comment'] if len(neg_df) > 0 else ''

    # Charts
    BG   = '#07090f'; CARD = '#131c2e'; POS  = '#10d97e'
    NEG  = '#ff4d6d'; NEU  = '#f59e0b'; MUTED= '#566a85'
    TEXT = '#dde6f0'; GRID = '#1c2a42'; ACC  = '#38bdf8'

    def fig64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return 'data:image/png;base64,' + b64

    def dark(ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5, linestyle='--')
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle='--')
        ax.set_axisbelow(True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG)
    bars = ax.bar(['Positive','Neutral','Negative'],
                  [pos_count, neu_count, neg_count],
                  color=[POS+'44', NEU+'44', NEG+'44'],
                  edgecolor=[POS, NEU, NEG], linewidth=2, width=0.45)
    for bar, val, clr in zip(bars, [pos_count, neu_count, neg_count], [POS, NEU, NEG]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(val), ha='center', color=clr, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(pos_count, neu_count, neg_count)*1.2)
    ax.tick_params(axis='x', colors=TEXT, labelsize=11)
    dark(ax, title='Sentiment Distribution', ylabel='Comments')
    img_bar = fig64(fig)

    # Donut chart
    sizes = [x for x in [pos_count, neu_count, neg_count] if x > 0]
    clrs  = [c for c, x in zip([POS+'55', NEU+'55', NEG+'55'],
                                 [pos_count, neu_count, neg_count]) if x > 0]
    edges = [c for c, x in zip([POS, NEU, NEG],
                                 [pos_count, neu_count, neg_count]) if x > 0]
    lbls  = [l for l, x in zip(['Positive','Neutral','Negative'],
                                 [pos_count, neu_count, neg_count]) if x > 0]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor=BG)
    _, _, autotexts = ax.pie(
        sizes, labels=lbls, colors=clrs, autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': BG, 'linewidth': 3, 'width': 0.55},
        textprops={'color': TEXT, 'fontsize': 10})
    for at, clr in zip(autotexts, edges):
        at.set_color(clr); at.set_fontweight('bold'); at.set_fontsize(11)
    ax.set_title('Sentiment Ratio', color=TEXT, fontsize=11, fontweight='bold', pad=14)
    img_pie = fig64(fig)

    # Aspect chart
    img_aspect = None
    if aspects:
        aspect_names = list(aspects.keys())
        pos_vals = [aspects[a]['pos'] for a in aspect_names]
        neg_vals = [aspects[a]['neg'] for a in aspect_names]
        fig, ax  = plt.subplots(figsize=(8, 3.5), facecolor=BG)
        x        = range(len(aspect_names))
        w        = 0.35
        ax.bar([i - w/2 for i in x], pos_vals, w, color=POS+'44', edgecolor=POS, linewidth=1.5, label='Positive')
        ax.bar([i + w/2 for i in x], neg_vals, w, color=NEG+'44', edgecolor=NEG, linewidth=1.5, label='Negative')
        ax.set_xticks(list(x)); ax.set_xticklabels(aspect_names, color=TEXT, fontsize=10)
        ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
        dark(ax, title='Aspect-Based Sentiment', ylabel='Comments')
        img_aspect = fig64(fig)

    # All table rows
    all_rows = []
    for _, row in df.iterrows():
        all_rows.append({
            'comment':    row['Comment'][:130] + ('...' if len(row['Comment']) > 130 else ''),
            'vader':      f"{row['VADER_Score']:+.4f}",
            'textblob':   f"{row['TextBlob_Score']:+.4f}",
            'score':      f"{row['Sentiment_Score']:+.4f}",
            'label':      row['Sentiment_Label'],
            'confidence': row['Confidence'],
            'sarcasm':    bool(row['Sarcasm']),
        })

    pos_kw = [{'word': w, 'count': c} for w, c in pos_words]
    neg_kw = [{'word': w, 'count': c} for w, c in neg_words]

    return {
        'total':            total,
        'pos_count':        pos_count,
        'neg_count':        neg_count,
        'neu_count':        neu_count,
        'sarc_count':       sarc_count,
        'pos_pct':          pos_pct,
        'neg_pct':          neg_pct,
        'neu_pct':          neu_pct,
        'top5_pos':         ', '.join(w for w, _ in pos_words[:5]),
        'top5_neg':         ', '.join(w for w, _ in neg_words[:5]),
        'pos_keywords':     pos_kw,
        'neg_keywords':     neg_kw,
        'img_bar':          img_bar,
        'img_pie':          img_pie,
        'img_aspect':       img_aspect,
        'aspects':          aspects,
        'top_pos_comment':  top_pos_comment,
        'top_neg_comment':  top_neg_comment,
        'all_rows':         all_rows,
        'nl_summary':       nl['summary_paras'],
        'nl_suggestions':   nl['suggestions'],
        'tone':             nl['tone'],
    }

# =============================================================================
# HTML PAGE
# =============================================================================
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Instagram Sentiment Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  /* ── THEME VARIABLES ── */
  :root{
    --bg:#07090f;--card:#131c2e;--border:#1c2a42;
    --pos:#10d97e;--neg:#ff4d6d;--neu:#f59e0b;--acc:#38bdf8;
    --text:#dde6f0;--muted:#566a85;--grid-line:rgba(56,189,248,.04);
    --shadow:rgba(0,0,0,.4);
  }

  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;
       line-height:1.65;overflow-x:hidden;min-height:100vh;transition:background .3s,color .3s}
  ::-webkit-scrollbar{width:5px}
  ::-webkit-scrollbar-track{background:var(--bg)}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

  body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
    background-image:linear-gradient(var(--grid-line) 1px,transparent 1px),
    linear-gradient(90deg,var(--grid-line) 1px,transparent 1px);
    background-size:64px 64px;animation:gr 28s linear infinite}
  @keyframes gr{0%{background-position:0 0,0 0}100%{background-position:64px 64px,64px 64px}}

  .blob{position:fixed;border-radius:50%;filter:blur(90px);pointer-events:none;
        z-index:0;animation:bl 9s ease-in-out infinite}
  .b1{width:480px;height:480px;background:rgba(56,189,248,.05);top:-80px;left:-100px}
  .b2{width:380px;height:380px;background:rgba(16,217,126,.05);bottom:-60px;right:-80px;animation-delay:4s}
  @keyframes bl{0%,100%{transform:scale(1)}50%{transform:scale(1.1)}}



  /* ── UPLOAD SCREEN ── */
  #upload-screen{position:relative;z-index:1;min-height:100vh;display:flex;
    flex-direction:column;justify-content:center;align-items:center;
    text-align:center;padding:3rem 2rem}
  .eye{display:inline-flex;align-items:center;gap:.5rem;padding:.35rem 1.1rem;
       border:1px solid var(--border);border-radius:999px;font-family:'DM Mono',monospace;
       font-size:.7rem;color:var(--acc);letter-spacing:.08em;margin-bottom:1.6rem;
       background:rgba(56,189,248,.06);animation:up .8s ease both}
  .dot{width:6px;height:6px;border-radius:50%;background:var(--acc);animation:bk 1.4s infinite}
  @keyframes bk{0%,100%{opacity:1}50%{opacity:.2}}
  h1{font-family:'Syne',sans-serif;font-size:clamp(2rem,5vw,3.8rem);font-weight:800;
     line-height:1.1;margin-bottom:1rem;
     background:linear-gradient(135deg,var(--text) 0%,var(--acc) 50%,var(--pos) 100%);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
     animation:up .9s .1s ease both}
  .hero-sub{font-size:1rem;color:var(--muted);max-width:540px;
            margin:0 auto 2.5rem;font-weight:300;animation:up 1s .2s ease both}
  @keyframes up{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}

  .dropzone{position:relative;width:100%;max-width:520px;
    border:2px dashed var(--border);border-radius:20px;padding:3rem 2rem;cursor:pointer;
    background:var(--card);transition:border-color .3s,background .3s,transform .2s;
    animation:up 1s .3s ease both}
  .dropzone:hover,.dropzone.drag{border-color:var(--acc);background:rgba(56,189,248,.05);transform:translateY(-4px)}
  .dropzone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .dz-icon{font-size:3rem;margin-bottom:1rem;display:block}
  .dz-title{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:.5rem;color:var(--text)}
  .dz-sub{font-size:.85rem;color:var(--muted)}

  #progress-box{display:none;margin-top:2rem;width:100%;max-width:520px;
    background:var(--card);border:1px solid var(--border);border-radius:14px;
    padding:1.5rem;animation:up .5s ease both}
  .prog-title{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;margin-bottom:1rem;color:var(--acc)}
  .prog-bar-wrap{background:var(--border);border-radius:999px;height:6px;overflow:hidden}
  .prog-bar{height:6px;background:linear-gradient(90deg,var(--acc),var(--pos));
            border-radius:999px;width:0%;transition:width .4s ease}
  #prog-status{font-size:.8rem;color:var(--muted);margin-top:.7rem;font-family:'DM Mono',monospace}
  #error-box{display:none;margin-top:1.5rem;padding:1rem 1.4rem;
    background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.3);
    border-radius:12px;color:var(--neg);font-size:.85rem;max-width:520px}

  /* ── RESULTS ── */
  #results-screen{display:none;position:relative;z-index:1}
  .res-header{padding:3rem 2rem 2rem;max-width:1100px;margin:0 auto;
    display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:1.5rem}
  .res-title{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
    background:linear-gradient(135deg,var(--text),var(--acc));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
  .res-sub{font-size:.85rem;color:var(--muted);margin-top:.25rem}

  .header-btns{display:flex;gap:.7rem;flex-wrap:wrap}
  .btn-new{padding:.6rem 1.4rem;border-radius:10px;border:1px solid var(--border);
    background:var(--card);color:var(--acc);font-family:'DM Mono',monospace;
    font-size:.78rem;cursor:pointer;transition:border-color .25s,transform .2s;letter-spacing:.04em}
  .btn-new:hover{border-color:var(--acc);transform:translateY(-2px)}
  .btn-dl{padding:.6rem 1.4rem;border-radius:10px;border:1px solid rgba(16,217,126,.4);
    background:rgba(16,217,126,.08);color:var(--pos);font-family:'DM Mono',monospace;
    font-size:.78rem;cursor:pointer;transition:border-color .25s,transform .2s;letter-spacing:.04em}
  .btn-dl:hover{border-color:var(--pos);transform:translateY(-2px)}

  .skipped-notice{display:none;padding:.6rem 1.2rem;max-width:1100px;margin:0 auto 1rem;
    background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
    border-radius:10px;font-size:.8rem;color:var(--neu);font-family:'DM Mono',monospace}

  /* Stat pills */
  .stats{display:flex;flex-wrap:wrap;gap:.9rem;padding:0 2rem 2rem;max-width:1100px;margin:0 auto}
  .sp{padding:.6rem 1.4rem;border-radius:12px;border:1px solid var(--border);
    background:var(--card);display:flex;flex-direction:column;align-items:center;
    flex:1;min-width:90px;transition:transform .25s,border-color .25s;
    box-shadow:0 2px 8px var(--shadow)}
  .sp:hover{transform:translateY(-3px);border-color:var(--acc)}
  .sn{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;line-height:1}
  .sl{font-size:.65rem;color:var(--muted);letter-spacing:.07em;margin-top:.2rem;font-family:'DM Mono',monospace}

  .rsec{padding:0 2rem 2.5rem;max-width:1100px;margin:0 auto}
  hr{border:none;border-top:1px solid var(--border);margin:0 2rem .5rem}
  .st{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;margin-bottom:.3rem;padding-top:2rem}
  .ss{color:var(--muted);font-size:.84rem;margin-bottom:1.6rem;font-style:italic}

  .cg{display:grid;gap:1.2rem;grid-template-columns:1fr 1fr;margin-bottom:1.2rem}
  @media(max-width:720px){.cg{grid-template-columns:1fr}}
  .cg1{grid-template-columns:1fr}
  .cb{background:var(--card);border:1px solid var(--border);border-radius:14px;
    padding:1.2rem;transition:border-color .25s,transform .25s;box-shadow:0 2px 8px var(--shadow)}
  .cb:hover{border-color:rgba(56,189,248,.3);transform:translateY(-3px)}
  .cl{font-family:'Syne',sans-serif;font-size:.83rem;font-weight:700;margin-bottom:.8rem;color:var(--text)}
  .cb img{width:100%;border-radius:8px;display:block}

  /* Top comment spotlight */
  .spotlight{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-bottom:1.2rem}
  @media(max-width:650px){.spotlight{grid-template-columns:1fr}}
  .spot-card{border-radius:14px;padding:1.2rem 1.4rem;border:1px solid transparent}
  .spot-pos{background:rgba(16,217,126,.07);border-color:rgba(16,217,126,.25)}
  .spot-neg{background:rgba(255,77,109,.07);border-color:rgba(255,77,109,.25)}
  .spot-label{font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.08em;margin-bottom:.5rem;opacity:.8}
  .spot-text{font-size:.9rem;line-height:1.6;color:var(--text)}
  .spot-score{margin-top:.6rem;font-family:'DM Mono',monospace;font-size:.75rem;opacity:.6}

  /* Keyword chips */
  .kg{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-bottom:1.2rem}
  @media(max-width:600px){.kg{grid-template-columns:1fr}}
  .kbox{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.2rem;box-shadow:0 2px 8px var(--shadow)}
  .ktitle{font-family:'Syne',sans-serif;font-size:.9rem;font-weight:700;margin-bottom:.9rem}
  .chips{display:flex;flex-wrap:wrap;gap:.4rem}
  .chip{display:inline-flex;align-items:center;gap:.35rem;padding:.28rem .75rem;
    border-radius:20px;font-family:'DM Mono',monospace;font-size:.72rem;
    border:1px solid transparent;transition:transform .2s;cursor:default}
  .chip:hover{transform:scale(1.07)}
  .chip-pos{background:rgba(16,217,126,.12);border-color:rgba(16,217,126,.3);color:var(--pos)}
  .chip-neg{background:rgba(255,77,109,.12);border-color:rgba(255,77,109,.3);color:var(--neg)}
  .chip-cnt{font-size:.62rem;opacity:.6}

  /* Search & filter bar */
  .filter-bar{display:flex;flex-wrap:wrap;gap:.7rem;margin-bottom:1.2rem;align-items:center}
  .search-input{flex:1;min-width:180px;padding:.5rem 1rem;border-radius:10px;
    border:1px solid var(--border);background:var(--card);color:var(--text);
    font-family:'DM Sans',sans-serif;font-size:.85rem;outline:none;
    transition:border-color .2s}
  .search-input:focus{border-color:var(--acc)}
  .filter-btns{display:flex;gap:.5rem;flex-wrap:wrap}
  .fbtn{padding:.4rem .9rem;border-radius:8px;border:1px solid var(--border);
    background:var(--card);color:var(--muted);font-family:'DM Mono',monospace;
    font-size:.72rem;cursor:pointer;transition:all .2s}
  .fbtn.active-all{border-color:var(--acc);color:var(--acc);background:rgba(56,189,248,.08)}
  .fbtn.active-pos{border-color:var(--pos);color:var(--pos);background:rgba(16,217,126,.08)}
  .fbtn.active-neu{border-color:var(--neu);color:var(--neu);background:rgba(245,158,11,.08)}
  .fbtn.active-neg{border-color:var(--neg);color:var(--neg);background:rgba(255,77,109,.08)}
  .fbtn.active-sarc{border-color:#a78bfa;color:#a78bfa;background:rgba(167,139,250,.08)}
  .result-count{font-size:.78rem;color:var(--muted);font-family:'DM Mono',monospace;margin-left:auto}

  /* Comment cards */
  .cards-wrap{display:flex;flex-direction:column;gap:.7rem}
  .ccard{border-radius:12px;padding:1rem 1.2rem;border-left:4px solid transparent;
    background:var(--card);border-top:1px solid var(--border);
    border-right:1px solid var(--border);border-bottom:1px solid var(--border);
    transition:transform .2s,box-shadow .2s}
  .ccard:hover{transform:translateX(4px);box-shadow:0 4px 16px var(--shadow)}
  .ccard-pos{border-left-color:var(--pos)}
  .ccard-neg{border-left-color:var(--neg)}
  .ccard-neu{border-left-color:var(--neu)}
  .ccard-top{display:flex;justify-content:space-between;align-items:flex-start;gap:.8rem;margin-bottom:.5rem}
  .ccard-text{font-size:.87rem;line-height:1.55;color:var(--text);flex:1}
  .ccard-badges{display:flex;gap:.4rem;flex-wrap:wrap;flex-shrink:0}
  .badge{display:inline-block;padding:.18rem .6rem;border-radius:6px;font-size:.68rem;
    font-family:'DM Mono',monospace;font-weight:500}
  .sarc-badge{display:inline-block;padding:.18rem .55rem;border-radius:6px;font-size:.65rem;
    font-family:'DM Mono',monospace;background:rgba(167,139,250,.15);
    border:1px solid rgba(167,139,250,.3);color:#a78bfa}
  .ccard-scores{display:flex;gap:1rem;margin-top:.5rem;flex-wrap:wrap}
  .score-item{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted)}
  .score-item span{color:var(--text)}
  .conf-low{font-size:.65rem;font-family:'DM Mono',monospace;color:var(--neu);
    background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.25);
    border-radius:5px;padding:.1rem .4rem}
  .conf-med{font-size:.65rem;font-family:'DM Mono',monospace;color:var(--acc);
    background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.2);
    border-radius:5px;padding:.1rem .4rem}
  .conf-high{font-size:.65rem;font-family:'DM Mono',monospace;color:var(--muted);
    background:rgba(86,106,133,.1);border:1px solid rgba(86,106,133,.2);
    border-radius:5px;padding:.1rem .4rem}

  .show-all-wrap{text-align:center;margin-top:1rem}
  .btn-show{padding:.5rem 1.4rem;border-radius:10px;border:1px solid var(--border);
    background:var(--card);color:var(--acc);font-family:'DM Mono',monospace;
    font-size:.76rem;cursor:pointer;transition:border-color .25s,transform .2s}
  .btn-show:hover{border-color:var(--acc);transform:translateY(-2px)}

  /* Summary */
  .sbox{background:var(--card);border:1px solid var(--border);border-radius:14px;
    padding:1.6rem;font-family:'DM Mono',monospace;font-size:.8rem;line-height:2;
    box-shadow:0 2px 8px var(--shadow)}
  .nl-para{font-size:.88rem;color:var(--text);line-height:1.8;margin-bottom:1rem;
    padding:.9rem 1.1rem;background:var(--card);border:1px solid var(--border);border-radius:10px}
  .nl-para:last-child{margin-bottom:0}
  .suggestions{display:flex;flex-direction:column;gap:.8rem;margin-top:1.2rem}
  .suggestion{display:flex;gap:.9rem;align-items:flex-start;background:var(--card);
    border:1px solid var(--border);border-radius:10px;padding:.9rem 1.1rem}
  .sug-icon{flex-shrink:0;font-size:1.1rem;margin-top:.1rem}
  .sug-text{font-size:.85rem;color:var(--muted);line-height:1.6}

  /* Disclaimer */
  .disclaimer{background:rgba(56,189,248,.05);border:1px solid rgba(56,189,248,.15);
    border-radius:14px;padding:1.4rem;margin-top:1.2rem}
  .disc-title{font-family:'Syne',sans-serif;font-size:.9rem;font-weight:700;
    color:var(--acc);margin-bottom:.8rem}
  .disc-item{display:flex;gap:.7rem;margin-bottom:.6rem;font-size:.83rem;color:var(--muted);line-height:1.5}
  .disc-item:last-child{margin-bottom:0}
  .disc-dot{width:6px;height:6px;border-radius:50%;background:var(--acc);
    flex-shrink:0;margin-top:.45rem}

  footer{position:relative;z-index:1;text-align:center;padding:2rem;
    border-top:1px solid var(--border);font-size:.76rem;color:var(--muted)}
  footer span{color:var(--acc)}

  @media(max-width:600px){
    .res-header{padding:2rem 1rem 1.5rem}
    .rsec,.stats{padding-left:1rem;padding-right:1rem}
    hr{margin-left:1rem;margin-right:1rem}
    .header-btns{width:100%}
    .btn-new,.btn-dl{flex:1;text-align:center}
    .ccard-top{flex-direction:column}
  }
</style>
</head>
<body>
<div class="blob b1"></div>
<div class="blob b2"></div>


<!-- UPLOAD SCREEN -->
<div id="upload-screen">
  <div class="eye"><span class="dot"></span> NLP &nbsp;&middot;&nbsp; SENTIMENT ANALYSIS</div>
  <h1>Instagram Comment<br/>Sentiment Analysis</h1>
  <p class="hero-sub">Upload your Instagram comments CSV. Each comment is analysed using VADER and TextBlob NLP models, then classified as Positive, Neutral, or Negative with aspect-level insights.</p>

  <div class="dropzone" id="dropzone">
    <input type="file" id="csv-input" accept=".csv"/>
    <span class="dz-icon">📂</span>
    <div class="dz-title">Drop your CSV file here</div>
    <div class="dz-sub">or click to browse and select</div>
  </div>

  <div id="progress-box">
    <div class="prog-title">⚙️ Analysing your comments...</div>
    <div class="prog-bar-wrap"><div class="prog-bar" id="prog-bar"></div></div>
    <div id="prog-status">Uploading file...</div>
  </div>
  <div id="error-box"></div>
</div>

<!-- RESULTS SCREEN -->
<div id="results-screen">
  <div class="res-header">
    <div>
      <div class="res-title">Analysis Results</div>
      <div class="res-sub" id="res-filename"></div>
    </div>
    <div class="header-btns">
      <button class="btn-dl" onclick="downloadCSV()">⬇ Download CSV</button>
      <button class="btn-new" onclick="resetPage()">⬅ New Analysis</button>
    </div>
  </div>

  <div id="skipped-notice" class="skipped-notice"></div>

  <!-- Stat Pills -->
  <div class="stats">
    <div class="sp"><span class="sn" id="s-total">—</span><span class="sl">COMMENTS</span></div>
    <div class="sp"><span class="sn" id="s-pos" style="color:var(--pos)">—</span><span class="sl">POSITIVE</span></div>
    <div class="sp"><span class="sn" id="s-neu" style="color:var(--neu)">—</span><span class="sl">NEUTRAL</span></div>
    <div class="sp"><span class="sn" id="s-neg" style="color:var(--neg)">—</span><span class="sl">NEGATIVE</span></div>
    <div class="sp"><span class="sn" id="s-pospct" style="color:var(--acc)">—</span><span class="sl">POS RATE</span></div>
    <div class="sp"><span class="sn" id="s-negpct" style="color:var(--acc)">—</span><span class="sl">NEG RATE</span></div>
    <div class="sp"><span class="sn" id="s-sarc" style="color:#a78bfa">—</span><span class="sl">SARCASTIC</span></div>
  </div>

  <hr/>

  <!-- Charts -->
  <div class="rsec">
    <div class="st">Sentiment Visualisation</div>
    <p class="ss">Visual breakdown of comment sentiment</p>
    <div class="cg">
      <div class="cb"><div class="cl">📊 Sentiment Distribution</div><img id="img-bar" src="" alt=""/></div>
      <div class="cb"><div class="cl">🥧 Sentiment Ratio</div><img id="img-pie" src="" alt=""/></div>
    </div>

  </div>

  <hr/>

  <!-- Aspect Sentiment -->
  <div class="rsec" id="aspect-section">
    <div class="st">Aspect-Based Sentiment</div>
    <p class="ss">Sentiment broken down by product aspects mentioned in comments</p>
    <div class="cg cg1">
      <div class="cb"><div class="cl">🔍 Aspect Analysis</div><img id="img-aspect" src="" alt=""/></div>
    </div>
  </div>

  <hr/>

  <!-- Spotlight -->
  <div class="rsec">
    <div class="st">Comment Spotlight</div>
    <p class="ss">The most positive and most negative comments from your dataset</p>
    <div class="spotlight">
      <div class="spot-card spot-pos">
        <div class="spot-label" style="color:var(--pos)">🌟 MOST POSITIVE</div>
        <div class="spot-text" id="spot-pos-text">—</div>
        <div class="spot-score" id="spot-pos-score"></div>
      </div>
      <div class="spot-card spot-neg">
        <div class="spot-label" style="color:var(--neg)">⚠️ MOST NEGATIVE</div>
        <div class="spot-text" id="spot-neg-text">—</div>
        <div class="spot-score" id="spot-neg-score"></div>
      </div>
    </div>
  </div>

  <hr/>

  <!-- Comment Cards with Filter -->
  <div class="rsec">
    <div class="st">All Comments</div>
    <p class="ss">Browse, search, and filter classified comments</p>
    <div class="filter-bar">
      <input class="search-input" id="search-input" type="text"
             placeholder="🔍 Search comments..." oninput="applyFilter()"/>
      <div class="filter-btns">
        <button class="fbtn active-all" id="fbtn-all"  onclick="setFilter('all')">All</button>
        <button class="fbtn" id="fbtn-pos"  onclick="setFilter('Positive')">✅ Positive</button>
        <button class="fbtn" id="fbtn-neu"  onclick="setFilter('Neutral')">➖ Neutral</button>
        <button class="fbtn" id="fbtn-neg"  onclick="setFilter('Negative')">❌ Negative</button>
        <button class="fbtn" id="fbtn-sarc" onclick="setFilter('sarcasm')">🙄 Sarcastic</button>
      </div>
      <span class="result-count" id="result-count"></span>
    </div>
    <div class="cards-wrap" id="cards-wrap"></div>
    <div class="show-all-wrap">
      <button class="btn-show" id="btn-show-all" onclick="toggleShowAll()">Show All Comments</button>
    </div>
  </div>

  <hr/>

  <!-- Summary -->
  <div class="rsec">
    <div class="st">Analysis Summary</div>
    <p class="ss">Final results at a glance</p>
    <div class="sbox" id="summary-box"></div>

    <div style="margin-top:2rem">
      <div class="st">Overall Sentiment Report</div>
      <p class="ss">What your customers are saying — in plain language</p>
      <div id="nl-summary-box"></div>
    </div>

    <div style="margin-top:2rem">
      <div class="st">Actionable Suggestions</div>
      <p class="ss">Steps to improve based on the sentiment analysis</p>
      <div class="suggestions" id="suggestions-box"></div>
    </div>

  </div>

  <footer>
    Customer Feedback Analysis of Instagram Comments Using NLP &nbsp;&middot;&nbsp;
    <span>Mini Project &mdash; S6 CS</span>
  </footer>
</div>

<script>
  const dropzone   = document.getElementById('dropzone');
  const csvInput   = document.getElementById('csv-input');
  const progBox    = document.getElementById('progress-box');
  const progBar    = document.getElementById('prog-bar');
  const progStatus = document.getElementById('prog-status');
  const errorBox   = document.getElementById('error-box');

  let globalData  = null;
  let showingAll  = false;
  let activeFilter= 'all';
  const PAGE_SIZE = 15;

  const STEPS = [
    [10, 'Reading your CSV file...'],
    [20, 'Detecting language of comments...'],
    [32, 'Converting emojis to text...'],
    [45, 'Tokenising & preprocessing with NLTK...'],
    [58, 'Running VADER sentiment scoring...'],
    [70, 'Running TextBlob sentiment scoring...'],
    [80, 'Combining scores & detecting sarcasm...'],
    [90, 'Analysing product aspects...'],
    [96, 'Generating charts & keywords...'],
  ];



  dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
  csvInput.addEventListener('change', () => {
    if (csvInput.files[0]) uploadFile(csvInput.files[0]);
  });

  async function uploadFile(file) {
    if (!file.name.endsWith('.csv')) { showError('Please upload a .csv file.'); return; }
    errorBox.style.display = 'none';
    progBox.style.display  = 'block';

    let stepIdx = 0;
    function nextStep() {
      if (stepIdx < STEPS.length) {
        const [pct, msg] = STEPS[stepIdx++];
        progBar.style.width    = pct + '%';
        progStatus.textContent = msg;
      }
    }
    nextStep();
    const stepTimer = setInterval(nextStep, 800);

    const formData = new FormData();
    formData.append('file', file);

    let data;
    try {
      const resp = await fetch('/analyse', { method: 'POST', body: formData });
      data = await resp.json();
    } catch(e) {
      clearInterval(stepTimer);
      showError('Connection error. Make sure mini.py is still running.'); return;
    }

    clearInterval(stepTimer);
    if (data.error) { showError(data.error); return; }

    progBar.style.width    = '100%';
    progStatus.textContent = 'Done!';
    await new Promise(r => setTimeout(r, 500));

    globalData = data;
    showResults(data, file.name);
  }

  function showError(msg) {
    errorBox.style.display = 'block';
    errorBox.textContent   = '⚠️  ' + msg;
    progBox.style.display  = 'none';
  }

  // Animated counter
  function animateCount(el, target, suffix='', duration=1000) {
    const start = performance.now();
    function step(now) {
      const p = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(ease * target) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function showResults(d, filename) {
    document.getElementById('upload-screen').style.display  = 'none';
    document.getElementById('results-screen').style.display = 'block';
    window.scrollTo(0, 0);

    document.getElementById('res-filename').textContent =
      filename + '  ·  ' + d.total + ' comments analysed';

    // Skipped notice
    const sn = document.getElementById('skipped-notice');
    let notices = [];
    if (d.skipped && d.skipped > 0)
      notices.push('⚠️  ' + d.skipped + ' non-English or too-short comment(s) were skipped.');
    if (d.sampled)
      notices.push('📊  File had more than 5,000 valid comments — a random sample of 5,000 was analysed.');
    if (notices.length > 0) {
      sn.style.display = 'block';
      sn.innerHTML = notices.join('<br>');
    } else { sn.style.display = 'none'; }

    // Animated stat pills
    animateCount(document.getElementById('s-total'),  d.total);
    animateCount(document.getElementById('s-pos'),    d.pos_count);
    animateCount(document.getElementById('s-neu'),    d.neu_count);
    animateCount(document.getElementById('s-neg'),    d.neg_count);
    animateCount(document.getElementById('s-pospct'), d.pos_pct, '%');
    animateCount(document.getElementById('s-negpct'), d.neg_pct, '%');
    animateCount(document.getElementById('s-sarc'),   d.sarc_count);

    // Charts
    document.getElementById('img-bar').src     = d.img_bar;
    document.getElementById('img-pie').src     = d.img_pie;

    if (d.img_aspect) {
      document.getElementById('img-aspect').src = d.img_aspect;
      document.getElementById('aspect-section').style.display = 'block';
    } else {
      document.getElementById('aspect-section').style.display = 'none';
    }

    // Spotlight
    document.getElementById('spot-pos-text').textContent  = d.top_pos_comment || 'No positive comments found.';
    document.getElementById('spot-neg-text').textContent  = d.top_neg_comment || 'No negative comments found.';

    // Cards
    activeFilter = 'all';
    showingAll   = false;
    setFilterUI('all');
    renderCards();

    // Summary
    const sep = '='.repeat(48);
    document.getElementById('summary-box').innerHTML = `
      <div style="color:var(--muted)">${sep}</div>
      <div style="color:var(--acc);font-weight:600">&nbsp; ANALYSIS SUMMARY</div>
      <div style="color:var(--muted)">${sep}</div>
      <div>&nbsp; Total Comments Analysed &nbsp;: <strong style="color:var(--text)">${d.total}</strong></div>
      <div>&nbsp; Positive Comments &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <strong style="color:var(--pos)">${d.pos_count} (${d.pos_pct}%)</strong></div>
      <div>&nbsp; Neutral Comments &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <strong style="color:var(--neu)">${d.neu_count} (${d.neu_pct}%)</strong></div>
      <div>&nbsp; Negative Comments &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <strong style="color:var(--neg)">${d.neg_count} (${d.neg_pct}%)</strong></div>
      <div>&nbsp; Possibly Sarcastic &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <strong style="color:#a78bfa">${d.sarc_count}</strong></div>
      <div style="color:var(--muted)">${sep}</div>`;

    document.getElementById('nl-summary-box').innerHTML =
      d.nl_summary.map(p => `<div class="nl-para">${p}</div>`).join('');

    const icons = ['&#128161;','&#128295;','&#128226;','&#129309;','&#11088;','&#128202;'];
    document.getElementById('suggestions-box').innerHTML =
      d.nl_suggestions.map((s, i) =>
        `<div class="suggestion">
           <span class="sug-icon">${icons[i % icons.length]}</span>
           <span class="sug-text">${s}</span>
         </div>`
      ).join('');
  }

  function getFilteredRows() {
    if (!globalData) return [];
    const query = document.getElementById('search-input').value.toLowerCase().trim();
    return globalData.all_rows.filter(r => {
      const matchFilter =
        activeFilter === 'all'    ? true :
        activeFilter === 'sarcasm'? r.sarcasm :
        r.label === activeFilter;
      const matchSearch = !query || r.comment.toLowerCase().includes(query);
      return matchFilter && matchSearch;
    });
  }

  function renderCards() {
    const rows    = getFilteredRows();
    const display = showingAll ? rows : rows.slice(0, PAGE_SIZE);
    document.getElementById('result-count').textContent = rows.length + ' comment' + (rows.length !== 1 ? 's' : '');

    document.getElementById('cards-wrap').innerHTML = display.map(r => {
      const clr     = r.label === 'Positive' ? 'pos' : r.label === 'Negative' ? 'neg' : 'neu';
      const confCls = r.confidence === 'High' ? 'conf-high' : r.confidence === 'Medium' ? 'conf-med' : 'conf-low';
      const labelClr= r.label === 'Positive' ? 'var(--pos)' : r.label === 'Negative' ? 'var(--neg)' : 'var(--neu)';
      const labelBg = r.label === 'Positive' ? '#10d97e18' : r.label === 'Negative' ? '#ff4d6d18' : '#f59e0b18';
      return `<div class="ccard ccard-${clr}">
        <div class="ccard-top">
          <div class="ccard-text">${r.comment}</div>
          <div class="ccard-badges">
            <span class="badge" style="color:${labelClr};background:${labelBg}">${r.label}</span>
            ${r.sarcasm ? '<span class="sarc-badge">🙄 Sarcastic?</span>' : ''}
            <span class="${confCls}">${r.confidence} conf.</span>
          </div>
        </div>
        <div class="ccard-scores">
          <span class="score-item">VADER <span>${r.vader}</span></span>
          <span class="score-item">TextBlob <span>${r.textblob}</span></span>
          <span class="score-item">Combined <span style="color:${labelClr}">${r.score}</span></span>
        </div>
      </div>`;
    }).join('') || '<div style="color:var(--muted);font-size:.85rem;padding:1rem 0">No comments match your filter.</div>';

    const btn = document.getElementById('btn-show-all');
    if (rows.length <= PAGE_SIZE) {
      btn.style.display = 'none';
    } else {
      btn.style.display = 'inline-block';
      btn.textContent   = showingAll ? 'Show Less' : 'Show All ' + rows.length + ' Comments';
    }
  }

  function setFilter(f) {
    activeFilter = f;
    showingAll   = false;
    setFilterUI(f);
    renderCards();
  }

  function setFilterUI(f) {
    const map = {all:'active-all', Positive:'active-pos', Neutral:'active-neu', Negative:'active-neg', sarcasm:'active-sarc'};
    ['all','pos','neu','neg','sarc'].forEach(id => {
      document.getElementById('fbtn-'+id).className = 'fbtn';
    });
    const key = f === 'Positive' ? 'pos' : f === 'Neutral' ? 'neu' : f === 'Negative' ? 'neg' : f === 'sarcasm' ? 'sarc' : 'all';
    document.getElementById('fbtn-'+key).className = 'fbtn ' + map[f];
  }

  function applyFilter() { showingAll = false; renderCards(); }

  function toggleShowAll() {
    showingAll = !showingAll;
    renderCards();
    if (!showingAll) document.getElementById('cards-wrap').scrollIntoView({behavior:'smooth'});
  }

  function downloadCSV() {
    if (!globalData) return;
    const rows   = globalData.all_rows;
    const header = ['Comment','VADER Score','TextBlob Score','Combined Score','Label','Confidence','Sarcasm Flag'];
    const lines  = [
      header.join(','),
      ...rows.map(r => [
        '"' + r.comment.replace(/"/g, '""') + '"',
        r.vader, r.textblob, r.score, r.label, r.confidence,
        r.sarcasm ? 'Yes' : 'No'
      ].join(','))
    ];
    const blob = new Blob([lines.join('\\n')], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'sentiment_results.csv'; a.click();
    URL.revokeObjectURL(url);
  }

  function resetPage() {
    document.getElementById('results-screen').style.display = 'none';
    document.getElementById('upload-screen').style.display  = 'flex';
    document.getElementById('progress-box').style.display   = 'none';
    document.getElementById('error-box').style.display      = 'none';
    document.getElementById('prog-bar').style.width         = '0%';
    document.getElementById('csv-input').value              = '';
    document.getElementById('skipped-notice').style.display = 'none';
    document.getElementById('search-input').value           = '';
    globalData = null; showingAll = false; activeFilter = 'all';
    window.scrollTo(0, 0);
  }
</script>
</body>
</html>"""

# =============================================================================
# FLASK SERVER
# =============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return render_template_string(PAGE)

@app.route('/analyse', methods=['POST'])
def analyse_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a .csv file.'})
    try:
        df_upload = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Could not read CSV: {str(e)}'})

    KNOWN_NAMES = {
        'comment','comments','comment text','commenttext','comment_text',
        'user comment','usercomment','user_comment','text','texts',
        'post text','posttext','post_text','review','reviews',
        'review text','reviewtext','review_text','user review','userreview',
        'customer review','customerreview','feedback','feedbacks',
        'user feedback','userfeedback','message','messages','caption',
        'captions','post','posts','content','body','description',
        'tweet','tweets','reply','replies','response','responses',
        'opinion','opinions','instagram comment','instagramcomment',
        'insta comment','customer comment','customercomment','product comment',
    }

    col = None
    for c in df_upload.columns:
        if c.strip().lower() in KNOWN_NAMES:
            col = c; break

    if col is None:
        best_col, best_score = None, 0
        for c in df_upload.columns:
            try:
                avg_len = df_upload[c].dropna().astype(str).apply(len).mean()
                if avg_len > best_score:
                    best_score = avg_len; best_col = c
            except Exception:
                continue
        if best_col and best_score >= 15:
            col = best_col

    if col is None:
        return jsonify({'error': 'Could not find a comments column in your CSV file.'})

    comments = df_upload[col].dropna().astype(str).tolist()
    if len(comments) == 0:
        return jsonify({'error': 'Your CSV file has no comments to analyse.'})

    comments = [c for c in comments if not is_irrelevant(c)]
    if len(comments) == 0:
        return jsonify({'error': 'No relevant comments found after filtering.'})

    english_comments = [c for c in comments if is_english(c)]
    skipped          = len(comments) - len(english_comments)
    if len(english_comments) == 0:
        return jsonify({'error': 'No valid English comments found in your file.'})

    comments = english_comments
    # FIX: sample instead of hard-rejecting large files
    sampled = False
    if len(comments) > 5000:
        import random; random.seed(42)
        comments = random.sample(comments, 5000)
        sampled  = True

    try:
        result = analyse(comments)
        result['skipped'] = skipped
        result['sampled'] = sampled
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

    return jsonify(result)

# =============================================================================
# LAUNCH
# =============================================================================
def open_browser():
    import time; time.sleep(1.2)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("=" * 55)
    print("  Instagram Comment Sentiment Analysis")
    print("=" * 55)
    print("  Opening browser at: http://127.0.0.1:5000")
    print("  Upload your CSV file on the webpage.")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 55 + "\n")
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, port=5000)
