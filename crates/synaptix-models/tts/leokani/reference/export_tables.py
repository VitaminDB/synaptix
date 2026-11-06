# Выгрузка констант-таблиц leokani_g2p.py в JSON для Rust-порта (src/g2p/tables.json).
import sys, json
sys.path.insert(0, '/tmp/claude-1000/-home-master-Storage---------2027-synaptix/8482c8ae-fa9d-4992-a010-08e84ae05ceb/scratchpad/pylib')
sys.path.insert(0, sys.argv[1])
import leokani_g2p as L
names = """ONES TENS ORDINAL_WORDS MONTHS MONTH_ABBR DAY_ABBR TITLE_ABBR STREET_ABBR WORD_ABBR UNITS CURRENCY SCALES
SCALE_WORDS ATTRIBUTIVE PER_UNIT VERSION_CUE ACRONYM_WORDS SHORTHAND SHORTHAND_CASED ZONES LETTER_CUE NUMBER_WORDS
MINUTE_NOUNS DIRECTIONS STREET_WORDS STATES TLDS TLD_SAY SEP_WORDS REGNAL ROMAN_CUE ROMAN_AMBIGUOUS SAY_AS SAY_AS_CASED
VERB_BEFORE ADJ_FORMS VERB_FORMS SUBJECTS OBJECTS CASED_SAY CODE_CUE ONSETS CODAS SENTENCE_START HG_CLASSES HOMOGRAPHS
TITLES ABBREV READ_PAST_BEFORE READ_PAST_AFTER LEAD_METAL WIND_VERB_BEFORE WIND_OBJECT LIVES_SUBJECT LIVES_OWNER
YEAR_BEFORE NAMES PRONOUN PRONOUN_AFTER DROPPED""".split()
out = {}
def conv(v):
    if isinstance(v, (set, frozenset)): return sorted(v)
    if isinstance(v, tuple): return [conv(x) for x in v]
    if isinstance(v, list): return [conv(x) for x in v]
    if isinstance(v, dict): return {k: conv(x) for k, x in v.items()}
    return v
for n in names:
    out[n] = conv(getattr(L, n))
out["VULGAR"] = L.Normalizer.VULGAR
out["VULGAR_WORDS"] = L.Normalizer.VULGAR_WORDS
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=False)
