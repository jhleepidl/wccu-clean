"""Original full-date-only sensitivity; does not replace official EM/F1."""
import datetime
from .common import score_answer
def date_key(text):
    for fmt in ('%d %B %Y', '%B %d, %Y', '%d %b %Y', '%b %d, %Y'):
        try:
            return datetime.datetime.strptime(text.strip().rstrip('.'), fmt).date().isoformat()
        except ValueError:
            pass
    return None

def quality(answer, answers):
    score = score_answer(answer, answers)
    key = date_key(answer)
    score['date_normalized_em'] = max(score['em'], float(bool(key) and any((date_key(a) == key for a in answers))))
    return score
