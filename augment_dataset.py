# augment_dataset.py
"""
Simple dataset augmentation for symptom -> disease CSV
Input expected: data/Diseases_Symptoms.csv with columns like: Code, Name, Symptoms, Treatments
Output: data/augmented_symptoms.csv (columns: symptoms,label)
This script creates synthetic variations using:
 - synonym replacement (WordNet)
 - adding modifiers (mild, severe)
 - shuffling word order
 - appending small contextual phrases
"""

import pandas as pd
import random
import re
import os
from nltk.corpus import wordnet

random.seed(42)

IN_PATH = "Diseases_Symptoms.csv"   # change if your file name differs
OUT_PATH = "data/augmented_symptoms.csv"
MIN_SAMPLES_PER_CLASS = 40   # target minimum number of examples per disease (tweakable)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def synonym_replace_once(text):
    """Replace one word with a WordNet synonym if available."""
    words = text.split()
    # pick candidate indices (words with length > 3)
    candidates = [i for i,w in enumerate(words) if len(w) > 3]
    if not candidates:
        return text
    idx = random.choice(candidates)
    word = words[idx]
    synsets = wordnet.synsets(word)
    if not synsets:
        return text
    # take first synset, get lemma names and choose one not equal to original
    lemmas = []
    for s in synsets:
        lemmas += s.lemma_names()
    lemmas = [l.replace('_',' ') for l in lemmas if l.lower() != word.lower()]
    lemmas = list(dict.fromkeys(lemmas))  # unique while preserving order
    if not lemmas:
        return text
    new_word = random.choice(lemmas)
    words[idx] = new_word
    return " ".join(words)

def add_modifier(text):
    """Add a modifier phrase to simulate severity or duration."""
    modifiers = [
        "mild", "moderate", "severe", "for a few days", "since this morning", 
        "intermittent", "constant", "worse at night", "worse during exertion"
    ]
    # randomly insert at end or start
    if random.random() < 0.6:
        return text + " " + random.choice(modifiers)
    else:
        return random.choice(modifiers) + " " + text

def shuffle_words(text):
    words = text.split()
    if len(words) <= 1:
        return text
    random.shuffle(words)
    return " ".join(words)

def append_context(text):
    contexts = [
        "noticed after exercise", "started after cold", "worse in morning", 
        "improved with rest", "no medication taken", "fever present", "no fever"
    ]
    if random.random() < 0.4:
        return text + ", " + random.choice(contexts)
    return text

def generate_variants(symptom_text, n_variants=4):
    """Generate a set of synthetic variants for one symptom string."""
    variants = set()
    base = clean_text(symptom_text)
    variants.add(base)
    attempts = 0
    while len(variants) < n_variants and attempts < n_variants * 8:
        attempts += 1
        choice = random.random()
        if choice < 0.3:
            v = synonym_replace_once(base)
        elif choice < 0.55:
            v = add_modifier(base)
        elif choice < 0.8:
            v = shuffle_words(base)
        else:
            v = append_context(base)
        v = clean_text(v)
        if v:
            variants.add(v)
    return list(variants)

def augment_dataframe(df, min_samples=MIN_SAMPLES_PER_CLASS):
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    # Expect 'name' as disease label and 'symptoms' column for text
    if 'name' not in df.columns or 'symptoms' not in df.columns:
        raise ValueError("CSV must have 'Name' and 'Symptoms' columns (case-insensitive).")
    df = df[['name','symptoms']].rename(columns={'name':'label','symptoms':'symptoms'})
    df['symptoms'] = df['symptoms'].astype(str).apply(clean_text)
    out_rows = []
    counts = df['label'].value_counts().to_dict()

    for label, group in df.groupby('label'):
        existing_texts = group['symptoms'].tolist()
        # keep existing examples
        for t in existing_texts:
            out_rows.append((t, label))
        current = counts.get(label, 0)
        # If class already has enough, optionally add 1-2 variants
        if current >= min_samples:
            # add 1-2 extra simple variants
            for t in existing_texts[:2]:
                variants = generate_variants(t, n_variants=2)
                for v in variants:
                    out_rows.append((v, label))
            continue
        # Need to synthesize until min_samples
        needed = min_samples - current
        # choose prototypes to augment (repeat as needed)
        prototypes = existing_texts if existing_texts else ["general pain"]
        proto_idx = 0
        while needed > 0:
            proto = prototypes[proto_idx % len(prototypes)]
            variants = generate_variants(proto, n_variants=min(6, needed+2))
            # add variants (avoid duplicates)
            for v in variants:
                if needed <= 0:
                    break
                out_rows.append((v, label))
                needed -= 1
            proto_idx += 1

    aug_df = pd.DataFrame(out_rows, columns=['symptoms','label'])
    # shuffle rows
    aug_df = aug_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return aug_df

def main():
    if not os.path.exists(IN_PATH):
        print("Input file not found:", IN_PATH)
        return
    df = pd.read_csv(IN_PATH)
    print("Loaded dataset rows:", len(df))
    # show original counts
    print("Original label counts (top 10):")
    print(df['Name'].value_counts().head(20))
    print("Augmenting... this may take a short while.")
    aug_df = augment_dataframe(df)
    print("Augmented dataset size:", len(aug_df))
    print("Augmented label counts (top 20):")
    print(aug_df['label'].value_counts().head(20))
    # ensure output folder exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    aug_df.to_csv(OUT_PATH, index=False)
    print("Saved augmented dataset to:", OUT_PATH)

if __name__ == "__main__":
    main()
