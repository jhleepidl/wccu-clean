# Function bodies extracted from StonyBrookNLP/ircot, commit 3c1820f698eea5eeddb4fba3c56b64c961e063e4
# Apache-2.0; see THIRD_PARTY_NOTICES.md and third_party/IRCoT_LICENSE.
import re
from typing import List

def is_reasoning_sentence(sentence: str) -> bool:
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True

    regex = re.compile(r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
    match = bool(re.match(regex, sentence))
    if match:
        return True

    return False

def remove_reasoning_sentences(sentences: List[str]) -> List[str]:
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]

def remove_wh_words(text: str) -> str:
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    text = " ".join(words)
    return text
