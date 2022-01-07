from os import path
eng_adj_path = path.join(path.dirname(__file__), 'english-adjectives.txt')
adverbs_path = path.join(path.dirname(__file__), 'adverbs.txt')

prepositions = [
'aboard',
'about',
'above',
'across',
'after',
'against',
'along',
'amid',
'among',
'anti',
'around',
'as',
'at',
'before',
'behind',
'below',
'beneath',
'beside',
'besides',
'between',
'beyond',
'but',
'by',
'concerning',
'considering',
'despite',
'down',
'during',
'except',
'excepting',
'excluding',
'following',
'for',
'from',
'in',
'inside',
'into',
'like',
'minus',
'near',
'of',
'off',
'on',
'onto',
'opposite',
'outside',
'over',
'past',
'per',
'plus',
'regarding',
'round',
'save',
'since',
'than',
'through',
'to',
'toward',
'towards',
'under',
'underneath',
'unlike',
'until',
'up',
'upon',
'versus',
'via',
'with',
'within',
'without',
]

found_invalid = [
    'and', 'of', 'in', 'to', ',', 'for', 'be', 'by', 'with', 'on', 'as', 'that', 'from', 'be', ')', '(', 'which',
    'at', 'be', 'be', 'be', ';', 'or', 'but', 'have', 'have', 'the', 'have', 'not', 'after', '"', 'include', 'also',
    'be', 'into', 'between', 'such', ':', 'do', 'while', 'when', 'during', 'would', 'over', 'since', '2019',
    'well', 'than', '2020', 'under', 'where', 'one', 'be', 'hold', '2018', 'can', 'through', '-',
    'make',  'out', 'there', 'know', 'due', 'a', 'take', 'up', 'begin', 'before', 'about',
    "'",  '4', '10', '3', '11', '&', '$', '12',  '2015', '2008','–', 'will',
    'so', 'do', 'follow', 'most', 'although', 'cause', 'only', '—',  '2007',  '2014', 'mostly', '5', 'say', '2017', '20',
    '2009',
]

invalid_relations = [
    'and', 'but', 'or', 'so', 'because', 'when', 'before', 'although', # conjunction
    'oh', 'wow', 'ouch', 'ah', 'oops',
    'what', 'how', 'where', 'when', 'who', 'whom',
    'a', 'and', 'the', 'there',
    'them', 'he', 'she', 'him', 'her', 'it', 'they',# pronoun
    'ten', 'hundred', 'thousand', 'million', 'billion',# unit
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',# number
    'year', 'month', 'day', 'daily',
] #+ found_invalid



auxiliaries = [
    'be', 'can', 'have', 'dare', 'may', 'will', 'would', 'should',
    'need', 'ought', 'shall', 'might', 'do', 'does', 'did',
    'be able to', 'had better','have to','need to','ought to','used to',
]

with open(eng_adj_path, 'r') as f:
    adjectives = [ line.strip().lower() for line in f]

with open(adverbs_path, 'r') as f:
    adverbs = [ line.strip().lower() for line in f]

# with open('corpus/Wordlist-Verbs-All.txt', 'r') as f:
#     verbs = [ line.strip().lower() for line in f]

invalid_relations += adjectives
invalid_relations += adverbs
invalid_relations += prepositions
# invalid_relations += verbs

invalid_relations_set = set(invalid_relations)
