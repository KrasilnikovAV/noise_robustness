"""Small, deterministic typo generator; callers own the random generator."""
import re

ERROR_TYPES = ('deletion', 'transposition', 'keyboard', 'duplication')
WORDS = re.compile(r'[^\W\d_]+', re.UNICODE)
# Nearby keys on a QWERTY keyboard.
NEIGHBORS = dict(zip('qwertyuiopasdfghjklzxcvbnm', (
    'wa', 'qeas', 'wrsd', 'etdf', 'ryfg', 'tugh', 'yihj', 'uojk', 'ipkl', 'ol',
    'qwsz', 'awedxz', 'serfcx', 'drtgvc', 'ftyhbv', 'gyujnb', 'huikmn',
    'jiolm', 'kop', 'asx', 'zsdc', 'xdfv', 'cfgb', 'vghn', 'bhjm', 'njk')))


def corrupt_text(text, p, rng, error_type=None):
    if not 0 <= p <= 1:
        raise ValueError('p must be between 0 and 1')
    if error_type is not None and error_type not in ERROR_TYPES:
        raise ValueError(f'Unknown error type: {error_type}')
    if p == 0:
        return text

    def replace(match):
        word = match.group()
        if len(word) < 4 or rng.random() >= p:
            return word
        kind = error_type or rng.choice(ERROR_TYPES)
        if kind == 'transposition':
            positions = [i for i in range(len(word) - 1) if word[i] != word[i + 1]]
            if not positions:
                return word
            i = rng.choice(positions)
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if kind == 'keyboard':
            positions = [i for i, char in enumerate(word) if char.lower() in NEIGHBORS]
            if not positions:
                return word
            i = rng.choice(positions)
            char = rng.choice(NEIGHBORS[word[i].lower()])
            if word[i].isupper():
                char = char.upper()
            return word[:i] + char + word[i + 1:]
        i = rng.randrange(len(word))
        if kind == 'deletion':
            return word[:i] + word[i + 1:]
        return word[:i] + word[i] + word[i:]

    return WORDS.sub(replace, text)
