import nltk
nltk.download('words')

common_attributes = set(['white','black','blue','green','red','brown','yellow',
'small','large','silver','wooden','gray','grey','metal','pink','tall',
'long','dark', 'color', 'right', 'left', 'backwards', 'front'])
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

# from spellchecker import SpellChecker
# spell = SpellChecker()

'''
Basic string cleaning so that we can map two variations of the same class to one
string (str): string to clean
'''
def clean_string(string):
    string = string.lower().strip().replace("_",' ').replace(' & ', ' and ')
    if len(string) >= 1 and string[-1] in ['.','?']:
        return string[:-1].strip()
    return string
    
'''
Essentially auto-correction
word (str): word to auto-correct
'''
def spellcorrect(word):
    if word in english_vocab:
        return word
    else:
        return spell.correction(word)

'''
Some hyphenated words are often just one unhyphenated word or two separate words.
This function tackles & and / as well to return words without these symbols if they are not necessary
word (str): word to remove symbols of
'''
def tackle_hyphens(word):
    if '-' in word:
        if ''.join(word.split('-')) in english_vocab:
            return [''.join(word.split('-'))]
        if all([subword in english_vocab for subword in word.split('-')]):
            return word.split('-')
        return [word]
    elif  '&' in word:
        if all([subword in english_vocab for subword in word.split('&')]):
            return word.split('&')
        return [word]
    elif  '/' in word:
        if all([subword in english_vocab for subword in word.split('/')]):
            return word.split('/')
        return [word]
    else:
        return [word]
    
'''
Cleans up object names
obj_name (str): name of object as indicated in Visual Genome dataset
'''
def clean_objects(obj_name):
    obj_name = clean_string(obj_name)
    obj_name_list = obj_name.split(" ")
    new_obj_name_list = []
    for word in obj_name_list:
        word_list = tackle_hyphens(word)
        for cleaned_word in word_list:
            # cleaned_word = spellcorrect(item, english_vocab, spell)
            if cleaned_word not in common_attributes:
                new_obj_name_list.append(cleaned_word)
    return ' '.join(new_obj_name_list)