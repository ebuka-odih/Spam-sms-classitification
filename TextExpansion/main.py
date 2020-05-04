import subprocess
import httplib, urllib, sys
import re
import sys
import os
from string import maketrans

BABELNET_SCRIPT = os.path.expanduser("~") + "/BabelNet/babelnet-api-1.0.1/run-babelnet.sh"


class EnglishDict:
    def __init__(self, dict_filename):
        f = open(dict_filename, "r")
        self.words = dict([line.strip("\n").lower().split("\t") for line in f.readlines()])
        f.close()

    def exists(self, word):
        lowered_word = word.lower()

        if self.words.has_key(lowered_word):
            return True
        return False

    def translate(self, word):
        lowered_word = word.lower()

        if self.words.has_key(lowered_word):
            return self.words[lowered_word]
        return word


class LingoToEnglish:
    def __init__(self, dict_filename):
        f = open(dict_filename, "r")
        self.words = dict([line.strip("\n").lower().split("\t") for line in f.readlines()])
        self.cache = {}
        f.close()

    def exists(self, word):
        lowered_word = word.lower()

        if self.words.has_key(lowered_word):
            return True
        return False

    def translate(self, word):
        lowered_word = word.lower()

        if self.words.has_key(lowered_word):
            return self.words[lowered_word]
        return word


english_dict = EnglishDict("dicts/englishDict.txt")
lingo_translator = LingoToEnglish("dicts/lingo.txt")
stopwords = []
all_concepts_cache = {}
wn_concepts_cache = {}
wiki_concepts_cache = {}


def load_stopwords(stopwords_file):
    f = open(stopwords_file, "r")
    stopwords.extend([stopword.strip("\n").lower() for stopword in f.readlines()])
    f.close()


def normalizeToken(token):
    wasChanged = False

    # Get the lingo translation for the token
    lingo_translated_token = lingo_translator.translate(token)

    # Splits the translation into subtokens
    subtokens = lingo_translated_token.split(" ")
    normalized_subtokens_list = []

    # For each subtoken, it gets its english "translation", and append it to the list of normalized subtokens
    for subtoken in subtokens:
        english_translated_subtoken = english_dict.translate(subtoken) + ' '
        normalized_subtokens_list.append(english_translated_subtoken)

    # Joins all the normalized subtokens, forming the normalization of the original token
    final_normalized_token = str().join(normalized_subtokens_list)

    # If the token has changed, it returns True, if the token remains the same, returns False
    if final_normalized_token != token:
        wasChanged = True

    return (wasChanged, final_normalized_token)


def get_wn_concepts(token):
    if wn_concepts_cache.has_key(token):
        return wn_concepts_cache[token]

    output = subprocess.check_output(["bash", BABELNET_SCRIPT, "-c -wn", token])
    divider = "\n"

    if output == "":
        concepts = [token]
    else:
        concepts = output[output.index(divider) + len(divider):-1].split("\n")
        if token.lower() in stopwords or (len(concepts) == 1 and concepts[0] == ""):
            concepts = [token]

        wn_concepts_cache[token] = concepts
    return concepts


def get_wiki_concepts(token):
    if wiki_concepts_cache.has_key(token):
        return wiki_concepts_cache[token]

    output = subprocess.check_output(["bash", BABELNET_SCRIPT, "-c -wiki", token])
    divider = "\n"

    if output == "":
        concepts = [token]
    else:
        concepts = output[output.index(divider) + len(divider):-1].split("\n")
        if token.lower() in stopwords or (len(concepts) == 1 and concepts[0] == ""):
            concepts = [token]

        wiki_concepts_cache[token] = concepts
    return concepts


def get_all_concepts(token):
    if all_concepts_cache.has_key(token):
        return all_concepts_cache[token]

    output = subprocess.check_output(["bash", BABELNET_SCRIPT, "-c", token])
    divider = "\n"

    if output == "":
        concepts = [token]
    else:
        concepts = output[output.index(divider) + len(divider):-1].split("\n")
        if token.lower() in stopwords or (len(concepts) == 1 and concepts[0] == ""):
            concepts = [token]

        all_concepts_cache[token] = concepts
    return concepts


def get_concepts(token, wn, wiki):
    if wn and not wiki:
        return get_wn_concepts(token)
    elif wiki and not wn:
        return get_wiki_concepts(token)
    else:
        return get_all_concepts(token)


def desambiguate_concepts(words, wn, wiki):
    if wn and not wiki:
        output = subprocess.check_output(["bash", BABELNET_SCRIPT, "-wn", words])
    elif wiki and not wn:
        output = subprocess.check_output(["bash", BABELNET_SCRIPT, "-wiki", words])
    else:
        output = subprocess.check_output(["bash", BABELNET_SCRIPT, words])

    concepts = output.split("\n")[:-1]

    concepts = [tuple(concept.split("\t")) for concept in concepts]

    if len(concepts) == 0:
        return {}

    if len(concepts[0]) == 1:
        return {}
    concepts_dict = dict(concepts)
    to_delete = []
    for key in concepts_dict.keys():
        if key.lower() in stopwords or key.lower() == concepts_dict[key]:
            del concepts_dict[key]

    return concepts_dict


def preProcessing(sample_list):
    postSamples = []
    emoticons = set([line.rstrip() for line in open('dicts/emoticons.dat')])
    lingos = set([line.rstrip() for line in open('dicts/lingo.txt')])

    BOW_PARSER = "$_-();,.:{}?[]#!@\'"
    TBL_PARSER = maketrans(BOW_PARSER, ' ' * len(BOW_PARSER))

    for sample in sample_list:
        tokens = sample.split(" ")

        for i, token in enumerate(tokens):
            if token not in emoticons and not lingo_translator.exists(token) and not english_dict.exists(token):
                tokens[i] = token.translate(TBL_PARSER)
                tokens[i] += ' '
            else:
                tokens[i] += ' '

        teste = str().join(tokens)

        msg = ''

        for token in re.split("[\t\ ]", teste):
            if token != '':
                msg += token
                msg += ' '

        if msg != '':
            postSamples.append(msg)

    return postSamples


def subtranslate(input_text, normalized_tokens, original_tokens, getConcepts, wn, wiki, desambiguation,
                 extra_dictionaries_paths, name):
    translated_input = ""
    load_stopwords("dicts/stopwords.txt")

    extra_dictionaries = []
    extra_dictionary = []

    for extra_dictionary_path in extra_dictionaries_paths:
        e = open(extra_dictionary_path, "r");
        extra_dictionary = dict([line.strip("\n").lower().split("\t") for line in e.readlines()])
        e.close()
        extra_dictionaries.append(extra_dictionary)

    # The preProcessing() function will return a list of the samples, without punctuation
    sms_list = []
    sms_list = preProcessing(input_text)

    translated_sms_list = []

    for sms in sms_list:
        sms_text = sms
        translated_sms_text = ""

        # from lingo to english
        for token in re.split("[\t\ ]", sms_text):
            if len(translated_sms_text) != 0:
                translated_sms_text += " "
            translated_sms_text += normalizeToken(token)[1]

        token_dict = {}

        if desambiguation:
            token_dict = desambiguate_concepts(translated_sms_text, wn, wiki)

        final_sms = ""
        for token in re.split("[\t\ ]", sms):

            originalToken = token
            search_token = normalizeToken(token)
            normalizedToken = search_token[1]

            subtokens = normalizedToken.split(" ")
            normalized_subtokens_list = []

            if original_tokens:
                final_sms += originalToken + " "

            for subtoken in subtokens:
                if subtoken != '':

                    if normalized_tokens and (subtoken != token or not original_tokens):
                        final_sms += subtoken + " "

                    if getConcepts:
                        tokenConcepts = ''

                        subtokenConcepts = " ".join(get_concepts(subtoken, wn, wiki))

                        if (subtokenConcepts == subtoken and (original_tokens or normalized_tokens)):
                            subtokenConcepts = ""
                        else:
                            tokenConcepts += subtokenConcepts
                            tokenConcepts.strip()
                            tokenConcepts += " "

                        final_sms += tokenConcepts

                    if desambiguation:
                        if token_dict.has_key(subtoken):
                            final_sms += token_dict[subtoken] + " "
                        elif not original_tokens and not normalized_tokens:
                            final_sms += subtoken + " "

            for new_dict in extra_dictionaries:
                if new_dict.has_key(token):
                    final_sms += new_dict[token] + " "

            if True not in (normalized_tokens, original_tokens, getConcepts, desambiguation):
                final_sms += token + " "

        final_sms = final_sms[:-1] + "\n";
        translated_input += final_sms

    save_file(translated_input, name)


def translate(input_text, translated_tokens, original_tokens, all_concepts, wn, wiki, desambiguation, All,
              extra_dictionaries_paths, name):
    if All:
        generate_all(input_text, wn, wiki, extra_dictionaries_paths, name)
    else:
        subtranslate(input_text, translated_tokens, original_tokens, all_concepts, wn, wiki, desambiguation,
                     extra_dictionaries_paths, name)


def generate_all(input_text, wn, wiki, extra_dictionaries_paths, name):
    conc = ""
    disa = ""
    norm = ""
    orig = ""
    ling_conc = ""
    ling_disa = ""
    orig_conc = ""
    orig_disa = ""
    orig_norm = ""
    orig_norm_conc = ""
    orig_norm_disa = ""

    translated_input = ""
    load_stopwords("dicts/stopwords.txt")

    extra_dictionaries = []
    extra_dictionary = []

    for extra_dictionary_path in extra_dictionaries_paths:
        e = open(extra_dictionary_path, "r");
        extra_dictionary = dict([line.strip("\n").lower().split("\t") for line in e.readlines()])
        e.close()
        extra_dictionaries.append(extra_dictionary)

    sms_list = []
    sms_list = preProcessing(input_text)

    translated_sms_list = []

    for sms in sms_list:
        sms_text = sms
        translated_sms_text = ""

        for token in re.split("[\t\ ]", sms_text):
            if len(translated_sms_text) != 0:
                translated_sms_text += " "
            translated_sms_text += normalizeToken(token)[1]

        token_dict = {}

        token_dict = desambiguate_concepts(translated_sms_text, wn, wiki)

        final_sms = ""

        for token in re.split("[\t\ ]", sms):

            originalToken = token
            search_token = normalizeToken(token)
            normalizedToken = search_token[1]

            subtokens = normalizedToken.split(" ")
            normalized_subtokens_list = []

            orig += token + " "
            orig_conc += token + " "
            orig_disa += token + " "
            orig_norm += token + " "
            orig_norm_conc += token + " "
            orig_norm_disa += token + " "

            for subtoken in subtokens:
                if subtoken != '':
                    norm += subtoken + " "
                    ling_conc += subtoken + " "
                    ling_disa += subtoken + " "

                    if (subtoken != token):
                        orig_norm += subtoken + " "
                        orig_norm_conc += subtoken + " "
                        orig_norm_disa += subtoken + " "

                    tokenConcepts = ''
                    if len(subtoken) < 2:
                        subtokenConcepts = subtoken
                    else:
                        subtokenConcepts = " ".join(get_concepts(subtoken, wn, wiki))
                    conc += subtokenConcepts + " "

                    if (subtokenConcepts != normalizedToken and subtokenConcepts != subtoken):
                        orig_norm_conc += subtokenConcepts + " "
                        ling_conc += subtokenConcepts + " "
                        orig_conc += subtokenConcepts + " "

                    if token_dict.has_key(subtoken):
                        orig_norm_disa += token_dict[subtoken] + " "
                        ling_disa += token_dict[subtoken] + " "
                        orig_disa += token_dict[subtoken] + " "
                        disa += token_dict[subtoken] + " "
                    else:
                        disa += subtoken + " "

            for new_dict in extra_dictionaries:
                if new_dict.has_key(token):
                    conc += new_dict[token] + " "
                    disa += new_dict[token] + " "
                    norm += new_dict[token] + " "
                    orig += new_dict[token] + " "
                    ling_conc += new_dict[token] + " "
                    ling_disa += new_dict[token] + " "
                    orig_conc += new_dict[token] + " "
                    orig_disa += new_dict[token] + " "
                    orig_norm += new_dict[token] + " "
                    orig_norm_conc += new_dict[token] + " "
                    orig_norm_disa += new_dict[token] + " "

        conc = conc[:-1] + '\n'
        disa = disa[:-1] + '\n'
        norm = norm[:-1] + '\n'
        orig = orig[:-2] + '\n'
        ling_conc = ling_conc[:-1] + '\n'
        ling_disa = ling_disa[:-1] + '\n'
        orig_conc = orig_conc[:-2] + '\n'
        orig_disa = orig_disa[:-2] + '\n'
        orig_norm = orig_norm[:-2] + '\n'
        orig_norm_conc = orig_norm_conc[:-2] + '\n'
        orig_norm_disa = orig_norm_disa[:-2] + '\n'

    save_file(conc, name + "_conc.txt")
    save_file(disa, name + "_disa.txt")
    save_file(norm, name + "_norm.txt")
    save_file(orig, name + "_orig.txt")
    save_file(ling_conc, name + "_norm_conc.txt")
    save_file(ling_disa, name + "_norm_disa.txt")
    save_file(orig_conc, name + "_orig_conc.txt")
    save_file(orig_disa, name + "_orig_disa.txt")
    save_file(orig_norm, name + "_orig_norm.txt")
    save_file(orig_norm_conc, name + "_orig_norm_conc.txt")
    save_file(orig_norm_disa, name + "_orig_norm_disa.txt")


def save_file(contents, name):
    f = open("files/" + str(name), "w")
    f.write(contents)
    f.close()
