import re
import string
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from spacy.matcher import PhraseMatcher

nlp = None
phrase_matcher = None
TERM_LEMMAS = None
RESOLUTION_LEMMAS = None

# Define term lists
term_dict = {
    'bilateral': ["bilateral", "widespread", "diffuse", "extensive", "multifocal", "bibasilar", "lungs", "bases", "lower lobes"],
    'opacity': ["consolidation", "consolidative", "opacity", "opacification", "density", "infiltrate", "pneumonia", "glass",
                      "ggo", "alveolar", "hazy", "consolidations", "infiltrates","atelectasis", "atelectatic","edema", "collapse",
                      "volume loss", "nodule", "mass", "nodule", "nodular", "tumor", "neoplasm", "mass", "granuloma", "nodules",
                      "lesion", "spiculated"],
    #Including opacity, consolidation both as they are included in definition of ARDS
    'consolidation': ["consolidation", "consolidative", "consolidations", "opacity", "opacification", "density", "densities","infiltrate", "glass",
                      "ggo",  "consolidations", "infiltrates"],
    'negation': ["no", "none", "not", "absent", "without", "free of"],
    'chronicity': ["chronic"],
    'effusion': ["effusion", "effusions", "pleural fluid"],
    'resolution': ["resolved", "resolution", "clear", "cleared", "absent", "not evident", "unremarkable", "removed", "removal", "normal", "not visualized",
                   "no longer visualized", "within normal limits", "resolve"],
    'edema': ["edema", "interstitial"],
    'atelectasis': ["atelectasis", "collapse", "volume loss", "atelectatic"],
    'pneumonia': ["pneumonia", "infection", "infectious", "pna"],
    'ptx': ["pneumothorax", "pneumothoraces", "ptx"],
    'cardiomegaly': ["cardiac", "cardiomegaly", "heart", "cardiomediastinal"],
    'enlarge': ["enlargement", "enlarge", "increased", "enlarged", "increase", "widen", "cardiomegaly", "big"],
    'mass': ["mass", "nodule", "nodular", "tumor", "neoplasm", "mass", "granuloma", "nodules", "lesion", "spiculated"],
    'device': ["pacer", "_line_", "lines", "picc", "ng tube", "og tube", "enteric tube", "valve", "catheter", "pacemaker", "hardware", "arthroplasty", "marker", "icd",
               "defib", "device", "drain", "plate", "screw", "cannula", "apparatus", "coil", "support", "equipment", "mediport", "port", "lead", "staple", "ekg", "telemetry",
               "enteral", "wire", "stent", "clip", "defibrillator", "nasogastric", "tubes", "drains", "devices"],
    'intubation': ["endotracheal", "et tube", "from the carina", "above the carina", "above carina", "et", "intubated", "intubation", 
                "endotracheal tube"],
    'heart_failure': ["chf", "heart failure", "volume overload", "cardiogenic", "vascular congestion", "congestion"],
    'fracture': ["fracture", "fx", "disruption", "nondisplaced", "fractures", "fractured"]
}

scoped_negation_phrases = [
    "no evidence of", "without evidence of", "free of", "no signs of"
]

NO_NEW_CONS_RE = re.compile(
    r'\bno new\b.*\b(?:' + '|'.join(map(re.escape, term_dict['opacity'])) + r')\b',
    re.IGNORECASE
)
LEFT_RE = re.compile(r'\bleft|retrocardiac\b', re.IGNORECASE)
RIGHT_RE = re.compile(r'\bright\b', re.IGNORECASE)
UPPER_RE = re.compile(r'\bupper\b', re.IGNORECASE)
LOWER_RE = re.compile(r'\blower\b', re.IGNORECASE)
RIGHT_CLEAR_RE = re.compile(r'\bright lung(s)? (is|are) (clear|unremarkable)', re.IGNORECASE)
LEFT_CLEAR_RE  = re.compile(r'\bleft lung(s)? (is|are) (clear|unremarkable)', re.IGNORECASE)

def set_nlp(model):
    # Build PhraseMatcher patterns using fully-processed docs
    phrase_matcher = PhraseMatcher(model.vocab, attr="LEMMA")

    # Flatten all phrases with their key so we can pipe them efficiently
    items = [(key, phrase) for key, phrases in term_dict.items() for phrase in phrases]
    texts = [t for _, t in items]

    # We want POS + lemmatizer, but we can skip NER and (optionally) parser for pattern creation
    # Keep tagger + attribute_ruler + lemmatizer ON.
    with model.select_pipes(disable=["ner", "parser"]):
        docs = list(model.pipe(texts, batch_size=256))

    # Add patterns grouped by key
    by_key = {}
    for (key, _), doc in zip(items, docs):
        by_key.setdefault(key, []).append(doc)

    for key, docs_for_key in by_key.items():
        phrase_matcher.add(key, docs_for_key)

    globals()["nlp"] = model
    globals()["phrase_matcher"] = phrase_matcher


def split_into_clauses_doc(doc):
    clauses = []
    start = 0

    for i, token in enumerate(doc):
        if token.text.lower() in {"and", "but"}:
            window = doc[i + 1:i + 6]
            has_verb = any(t.pos_ in {"VERB", "AUX"} for t in window)
            has_subject = any(t.dep_ in {"nsubj", "nsubjpass", "expl"} for t in window)

            prev_context = doc[max(0, i - 6):i].text.lower()
            if any(neg_phrase in prev_context for neg_phrase in scoped_negation_phrases):
                continue

            if has_verb and has_subject:
                span = doc[start:i]
                if span.text.strip():
                    clauses.append(span.text.strip())
                start = i + 1

    final = doc[start:]
    if final.text.strip():
        clauses.append(final.text.strip())

    return clauses or [doc.text]


"""
# Create PhraseMatcher patterns
for key, phrases in term_dict.items():
    phrase_matcher.add(key, [nlp(p) for p in phrases])
"""

no_new_consolidation_pattern = re.compile(r'\bno new\b.*\b(?:' + '|'.join(term_dict['opacity']) + r')\b', re.IGNORECASE)

def calc_numeric_flag(present, negation):
    if present:
        return 1 if not negation else -1
    return 0
    
def build_lemma_caches():
    global TERM_LEMMAS, RESOLUTION_LEMMAS
    # Build lemma sets once, using the initialized nlp pipeline
    TERM_LEMMAS = {
        key: {tok.lemma_.lower() for phrase in phrases for tok in nlp(phrase)}
        for key, phrases in term_dict.items()
    }
    RESOLUTION_LEMMAS = TERM_LEMMAS["resolution"]

def clause_has_resolution_context(doc, key_terms, resolution_terms, negation_terms):
    """
    key_terms / resolution_terms / negation_terms are the LISTS you pass in:
      clause_has_resolution_context(doc, term_dict[term], term_dict['resolution'], term_dict['negation'])
    We map those lists back to the corresponding term_dict key (if possible) so we can use cached lemma sets.
    """

    # Build a reverse lookup from list object identity -> term_dict key
    # (done once, cached on the function object)
    if not hasattr(clause_has_resolution_context, "_id_to_key"):
        clause_has_resolution_context._id_to_key = {id(v): k for k, v in term_dict.items()}

    id_to_key = clause_has_resolution_context._id_to_key

    # Use cached lemma sets when we can; otherwise fall back (should be rare)
    kt = id_to_key.get(id(key_terms))
    rt = id_to_key.get(id(resolution_terms))

    if kt is not None:
        key_lemmas = TERM_LEMMAS[kt]
    else:
        key_lemmas = {tok.lemma_.lower() for phrase in key_terms for tok in nlp(phrase)}

    if rt is not None:
        resolution_lemmas = TERM_LEMMAS[rt]
    else:
        resolution_lemmas = {tok.lemma_.lower() for phrase in resolution_terms for tok in nlp(phrase)}

    # Catch explicit 'within normal limits' even if dependency tree is complex
    if "within normal limits" in doc.text.lower():
        for token in doc:
            if token.lemma_.lower() in key_lemmas:
                return True

    for token in doc:
        # Case 1: resolution word's head is a key term
        if token.lemma_.lower() in resolution_lemmas:
            head = token.head
            if head.lemma_.lower() in key_lemmas:
                return True
            for child in head.children:
                if child.lemma_.lower() in key_lemmas:
                    return True

        # Case 2: key term's head is a resolution word
        if token.lemma_.lower() in key_lemmas:
            head = token.head
            if head.lemma_.lower() in resolution_lemmas:
                return True
            for child in head.children:
                if child.lemma_.lower() in resolution_lemmas:
                    return True

        # Case 3: copula-based construction — 'heart is normal'
        if token.dep_ == "nsubj" and token.lemma_.lower() in key_lemmas:
            for sibling in token.head.children:
                if sibling.dep_ in {"acomp", "attr"} and sibling.lemma_.lower() in resolution_lemmas:
                    return True

        # Case 4: key term is the subject of a resolution verb
        if token.dep_ == "nsubj" and token.lemma_.lower() in key_lemmas:
            head = token.head
            if head.lemma_.lower() in resolution_lemmas:
                return True

        # Case 5: loose fallback - resolution and key terms co-occur in clause
        doc_lemmas = {t.lemma_.lower() for t in doc}
        if key_lemmas & doc_lemmas and resolution_lemmas & doc_lemmas:
            return True

    return False


def check_normal_lung_language(doc):
    sl = doc.text.lower()
    return bool(re.search(r'\blung(s)?\b', sl) and re.search(r'\b(clear|unremarkable)\b', sl))

def consolidation_modified_by_bilateral(doc, matches):
    consolidation_spans = []
    bilat_spans = []

    def has_lobe_with_laterality(doc):
        for token in doc:
            if token.text.lower() in {"lobe", "lobes"}:
                related = list(token.children) + list(token.ancestors)
                related_texts = [t.text.lower() for t in related]
                if "left" in related_texts or "right" in related_texts:
                    return True
        return False

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        if label == 'consolidation':
            consolidation_spans.append(span)
        elif label == 'bilateral':
            span_text = span.text.lower()
            if span_text in {"lower lobes", "upper lobes", "lower lobe", "upper lobe"}:
                if has_lobe_with_laterality(doc):
                    continue
            bilat_spans.append(span)

    for cons in consolidation_spans:
        for bilat in bilat_spans:
            # Check if the bilat phrase is within 8 tokens of cons
            if abs(cons.start - bilat.start) <= 8:
                return True

            # Check if any tokens are connected via dependency path
            for cons_token in cons:
                for bilat_token in bilat:
                    path = cons_token.subtree
                    if bilat_token in path:
                        return True
                    if bilat_token in cons_token.children or bilat_token in cons_token.ancestors:
                        return True
    return False

def is_scoped_negation(doc, key_terms):
    key_terms = set(term.lower() for term in key_terms)

    # Catch scoped negation from tokens
    for token in doc:
        if token.text.lower() in {"no", "without", "free"}:
            head = token.head
            if head.pos_ not in {"NOUN", "PROPN"}:
                head = token

            if head.lemma_.lower() in {"change", "changes", "alteration", "shift", "difference"}:
                continue

            subtree = list(head.subtree)
            scope_lemmas = set(t.lemma_.lower() for t in subtree)

            if key_terms & scope_lemmas:
                return True

            # Handle coordination (e.g. no X, Y, or Z)
            after = [t for t in doc if t.i > token.i]
            after_lemmas = set(t.lemma_.lower() for t in after[:20])
            if key_terms & after_lemmas:
                return True

    # Phrase-based fallback
    for phrase in scoped_negation_phrases:
        if phrase in doc.text.lower():
            tail = doc.text.lower().split(phrase, 1)[-1]
            if any(re.search(r'\b{}\b'.format(re.escape(term)), tail) for term in key_terms):
                return True

    # New: handle "no...to suggest X, Y, or Z" or "no X to suggest Y"
    text = doc.text.lower()
    if text.startswith("no") or "no " in text:
        if "suggest" in text or "evidence" in text:
            tail = text.split("no", 1)[-1]
            for term in key_terms:
                if re.search(r'\b{}\b'.format(re.escape(term)), tail):
                    return True

    return False

def get_cardiomegaly_flag(doc, cmg_mention, enl_mention):
    present = cmg_mention and enl_mention  # Require both cardiac and enlargement mention
    negated = (
        is_scoped_negation(doc, term_dict['cardiomegaly']) or
        is_scoped_negation(doc, term_dict['enlarge']) or
        clause_has_resolution_context(doc, term_dict['cardiomegaly'], term_dict['resolution'], term_dict['negation']) or
        clause_has_resolution_context(doc, term_dict['enlarge'], term_dict['resolution'], term_dict['negation'])
    )
    return calc_numeric_flag(present, negated) if present else -1 if negated else 0

# Consolidates detected terms and applies laterality logic to consolidation and atelectasis
def detect_bilateral_consolidation(text):
    doc = nlp(text)
    matches = phrase_matcher(doc)
    matched_labels = {nlp.vocab.strings[m_id] for m_id, start, end in matches}

    has_term = lambda label: label in matched_labels

    cons_present = has_term('consolidation')
    opac_present = has_term('opacity')
    bilat_present = has_term('bilateral')
    hf_present = has_term('heart_failure')
    ed_present = has_term('edema')
    eff_present = has_term('effusion')
    at_present = has_term('atelectasis')
    pn_present = has_term('pneumonia')
    cmg_mention = has_term('cardiomegaly')
    enl_mention = has_term('enlarge')
    mass_present = has_term('mass')
    dev_present = has_term('device')
    ptx_present = has_term('ptx')
    et_present = has_term('intubation')
    fx_present = has_term('fracture')

    neg = lambda term: (
        clause_has_resolution_context(doc, term_dict[term], term_dict['resolution'], term_dict['negation']) or
        is_scoped_negation(doc, term_dict[term])
    )

    has_left = bool(LEFT_RE.search(text))
    has_right = bool(RIGHT_RE.search(text))
    has_upper = bool(UPPER_RE.search(text))
    has_lower = bool(LOWER_RE.search(text))

    normal_lungs = check_normal_lung_language(doc)

    cons_num = 0 if NO_NEW_CONS_RE.search(text) else calc_numeric_flag(cons_present, neg('consolidation'))
    if normal_lungs and not (has_left or has_right or has_upper or has_lower):
        cons_num = -1

    left_num = calc_numeric_flag(cons_present and has_left, neg('consolidation'))
    right_num = calc_numeric_flag(cons_present and has_right, neg('consolidation'))
    if RIGHT_CLEAR_RE.search(text):
        right_num = -1
    if LEFT_CLEAR_RE.search(text):
        left_num = -1

    # Consolidation bilaterality check
    bilat_num = 0
    if cons_num == 1:
        if left_num == 1 and right_num == 1:
            bilat_num = 1
        elif (left_num == 1 and right_num == -1) or (right_num == 1 and left_num == -1):
            bilat_num = -1
        elif consolidation_modified_by_bilateral(doc, matches):
            bilat_num = 1
    elif cons_num == -1 or (normal_lungs and not (has_left or has_right or has_upper or has_lower)):
        bilat_num = -1

    # Atelectasis bilaterality check
    at_bilat_num = 0
    if at_present and not (has_left and not has_right) and not (has_right and not has_left):
        if consolidation_modified_by_bilateral(doc, matches):
            at_bilat_num = 1
        elif bilat_present:
            at_bilat_num = 1 if not neg('bilateral') else -1
    elif not has_left and not has_right:
        # Also allow bilateral atelectasis negation to propagate if truly generic
        if neg('atelectasis') and neg('bilateral'):
            at_bilat_num = -1

    cmg_num = get_cardiomegaly_flag(doc, cmg_mention, enl_mention)

    return {
        'any_opacity': -1 if normal_lungs and not (
                    has_left or has_right or has_upper or has_lower) else calc_numeric_flag(opac_present, neg('opacity')),
        'consolidation': cons_num,
        'left_cons': left_num,
        'right_cons': right_num,
        'bilateral_consolidation': bilat_num,
        'edema': calc_numeric_flag(ed_present, neg('edema')),
        'atelectasis': calc_numeric_flag(at_present, neg('atelectasis')),
        'r_atelectasis': calc_numeric_flag(at_present and has_right, neg('atelectasis')),
        'l_atelectasis': calc_numeric_flag(at_present and has_left, neg('atelectasis')),
        'bi_atelectasis': at_bilat_num,
        'heart_failure': calc_numeric_flag(hf_present, neg('heart_failure')),
        'effusion': calc_numeric_flag(eff_present, neg('effusion')),
        'pneumonia': calc_numeric_flag(pn_present, neg('pneumonia')),
        'pneumothorax': calc_numeric_flag(ptx_present, neg('ptx')),
        'cardiomegaly': cmg_num,
        'mass': calc_numeric_flag(mass_present, neg('mass')),
        'devices': calc_numeric_flag(dev_present, neg('device')),
        'intubation': calc_numeric_flag(et_present, neg('intubation')),
        'fracture': calc_numeric_flag(ed_present, neg('fracture'))
    }    
    
def detect_bilateral_consolidation_doc(doc):
    text = doc.text
    matches = phrase_matcher(doc)
    matched_labels = {nlp.vocab.strings[m_id] for m_id, start, end in matches}

    has_term = lambda label: label in matched_labels

    cons_present = has_term('consolidation')
    opac_present = has_term('opacity')
    bilat_present = has_term('bilateral')
    hf_present = has_term('heart_failure')
    ed_present = has_term('edema')
    eff_present = has_term('effusion')
    at_present = has_term('atelectasis')
    pn_present = has_term('pneumonia')
    cmg_mention = has_term('cardiomegaly')
    enl_mention = has_term('enlarge')
    mass_present = has_term('mass')
    dev_present = has_term('device')
    ptx_present = has_term('ptx')
    et_present = has_term('intubation')
    fx_present = has_term('fracture')

    neg = lambda term: (
        clause_has_resolution_context(doc, term_dict[term], term_dict['resolution'], term_dict['negation']) or
        is_scoped_negation(doc, term_dict[term])
    )

    has_left  = bool(re.search(r'\bleft|retrocardiac\b', text, re.IGNORECASE))
    has_right = bool(re.search(r'\bright\b', text, re.IGNORECASE))
    has_upper = bool(re.search(r'\bupper\b', text, re.IGNORECASE))
    has_lower = bool(re.search(r'\blower\b', text, re.IGNORECASE))
    normal_lungs = check_normal_lung_language(doc)

    cons_num = 0 if NO_NEW_CONS_RE.search(text) else calc_numeric_flag(cons_present, neg('consolidation'))
    if normal_lungs and not (has_left or has_right or has_upper or has_lower):
        cons_num = -1

    left_num = calc_numeric_flag(cons_present and has_left, neg('consolidation'))
    right_num = calc_numeric_flag(cons_present and has_right, neg('consolidation'))
    if re.search(r'\bright lung(s)? (is|are) (clear|unremarkable)', text.lower()):
        right_num = -1
    if re.search(r'\bleft lung(s)? (is|are) (clear|unremarkable)', text.lower()):
        left_num = -1

    bilat_num = 0
    if cons_num == 1:
        if left_num == 1 and right_num == 1:
            bilat_num = 1
        elif (left_num == 1 and right_num == -1) or (right_num == 1 and left_num == -1):
            bilat_num = -1
        elif consolidation_modified_by_bilateral(doc, matches):
            bilat_num = 1
    elif cons_num == -1 or (normal_lungs and not (has_left or has_right or has_upper or has_lower)):
        bilat_num = -1

    at_bilat_num = 0
    if at_present and not (has_left and not has_right) and not (has_right and not has_left):
        if consolidation_modified_by_bilateral(doc, matches):
            at_bilat_num = 1
        elif bilat_present:
            at_bilat_num = 1 if not neg('bilateral') else -1
    elif not has_left and not has_right:
        if neg('atelectasis') and neg('bilateral'):
            at_bilat_num = -1

    cmg_num = get_cardiomegaly_flag(doc, cmg_mention, enl_mention)

    return {
        'any_opacity': -1 if normal_lungs and not (has_left or has_right or has_upper or has_lower)
                      else calc_numeric_flag(opac_present, neg('opacity')),
        'consolidation': cons_num,
        'left_cons': left_num,
        'right_cons': right_num,
        'bilateral_consolidation': bilat_num,
        'edema': calc_numeric_flag(ed_present, neg('edema')),
        'atelectasis': calc_numeric_flag(at_present, neg('atelectasis')),
        'r_atelectasis': calc_numeric_flag(at_present and has_right, neg('atelectasis')),
        'l_atelectasis': calc_numeric_flag(at_present and has_left, neg('atelectasis')),
        'bi_atelectasis': at_bilat_num,
        'heart_failure': calc_numeric_flag(hf_present, neg('heart_failure')),
        'effusion': calc_numeric_flag(eff_present, neg('effusion')),
        'pneumonia': calc_numeric_flag(pn_present, neg('pneumonia')),
        'pneumothorax': calc_numeric_flag(ptx_present, neg('ptx')),
        'cardiomegaly': cmg_num,
        'mass': calc_numeric_flag(mass_present, neg('mass')),
        'devices': calc_numeric_flag(dev_present, neg('device')),
        'intubation': calc_numeric_flag(et_present, neg('intubation')),
        'fracture': calc_numeric_flag(fx_present, neg('fracture')),
    }

def side_mentions_consolidation(clause, side):
    """Check if 'left' or 'right' is meaningfully associated with consolidation-related terms in the clause."""
    side_terms = {
        "left": ["left", "retrocardiac"],
        "right": ["right"]
    }
    cons_terms = term_dict['consolidation']
    clause_lower = clause.lower()

    for s_term in side_terms[side]:
        for c_term in cons_terms:
            # Check if both terms occur in proximity
            if re.search(r'\b{}\b.*\b{}\b'.format(s_term, c_term), clause_lower) or \
               re.search(r'\b{}\b.*\b{}\b'.format(c_term, s_term), clause_lower):
                return True
    return False

def detect_labels_for_sentences(sentences):
    all_sentence_labels = []
    for sent in sentences:
        clause_labels = []

        sent_doc = nlp(sent)                       # parse sentence once
        clauses = split_into_clauses_doc(sent_doc) # no nlp() inside

        for clause_text in clauses:
            clause_doc = nlp(clause_text)          # parse each clause once
            label = detect_bilateral_consolidation_doc(clause_doc)

            if side_mentions_consolidation(clause_text, "left"):
                if label['consolidation'] == 1:
                    label['left_cons'] = 1
                elif label['consolidation'] == -1:
                    label['left_cons'] = -1

            if side_mentions_consolidation(clause_text, "right"):
                if label['consolidation'] == 1:
                    label['right_cons'] = 1
                elif label['consolidation'] == -1:
                    label['right_cons'] = -1

            clause_labels.append(label)

        all_sentence_labels.append(aggregate_labels(clause_labels))

    return all_sentence_labels

    
def aggregate_labels(labels_list):
    if not labels_list:
        return {}
    keys = ['any_opacity','consolidation','left_cons','right_cons','bilateral_consolidation','pneumonia','edema','atelectasis','r_atelectasis',
            'l_atelectasis','bi_atelectasis','mass','pneumothorax','effusion','heart_failure','cardiomegaly','devices','intubation','fracture']
    agg = {}
    for k in keys:
        total = sum(lbl.get(k,0) for lbl in labels_list)
        agg[k] = 1 if total > 0 else -1 if total < 0 else 0

    if agg['left_cons'] == 1 and agg['right_cons'] == 1:
        agg['bilateral_consolidation'] = 1
    elif agg['left_cons'] == 1 and agg['right_cons'] == -1:
        agg['bilateral_consolidation'] = -1
    elif agg['left_cons'] == -1 and agg['right_cons'] == 1:
        agg['bilateral_consolidation'] = -1
    elif agg['any_opacity'] == -1:
        agg['bilateral_consolidation'] = -1
    if agg['l_atelectasis'] == 1 and agg['r_atelectasis'] == 1:
        agg['bi_atelectasis'] = 1

    return agg

def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    return " ".join([token.text if token.text.lower() in ["left","right","lungs", "bases","lower lobes"] else token.lemma_ for token in doc])

def lemmatize_sentences(sentences):
    return [lemmatize_sentence(s) for s in sentences]

def process_text(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    per_lbls = detect_labels_for_sentences(sentences)
    agg_lbls = aggregate_labels(per_lbls)
    return sentences, per_lbls, agg_lbls

def init(model_name: str = "en_core_web_sm"):
    model = spacy.load(model_name)

    # HARD FAIL if POS pipeline is missing
    required = {"tagger", "morphologizer"}
    if not any(p in model.pipe_names for p in required):
        raise RuntimeError(
            f"spaCy pipeline missing POS component. pipe_names={model.pipe_names}"
        )

    set_nlp(model)
    build_lemma_caches()
    return model

def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run CXR labeler on a text string or .txt file.")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model name to load")
    parser.add_argument("--text", help="Text to label")
    parser.add_argument("--infile", help="Path to input .txt file")
    parser.add_argument("--outfile", help="Path to output JSON file (optional)")
    args, _ = parser.parse_known_args()

    if not args.text and not args.infile:
        parser.error("Provide either --text or --infile")

    if args.infile:
        text = Path(args.infile).read_text(encoding="utf-8")
    else:
        text = args.text

    init(args.model)  # <-- THIS is the missing "entry/init" step
    
    #Debug Text
    print("PIPE:", nlp.pipe_names)
    d1 = nlp("endotracheal tube unchanged")
    print("POS_OK:", all(t.pos_ != "" for t in d1), "LEMMA:", [t.lemma_ for t in d1])
    
    sentences, per_sentence_labels, aggregate = process_text(text)

    out = {
        "sentences": sentences,
        "per_sentence_labels": per_sentence_labels,
        "aggregate_labels": aggregate,
    }

    if args.outfile:
        Path(args.outfile).write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
