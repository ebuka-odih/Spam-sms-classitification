TextExpansion v.1
Short text messages (e.g. posts in blogs, forums, social networks, etc) represent a challenging problem for traditional learning methods nowadays, since such messages are usually fairly short and normally rife of slangs, idioms, symbols and acronyms that make even tokenization a difficult task. In this scenario, we have designed the TextExpansion tool which aims to normalize and expand the original short and messy text messages in order to acquire better attributes and enhance the classification/clustering performance.

The proposed approach is based on lexicography and semantic dictionaries along with state-of-the-art techniques for semantic analysis and context detection. This technique is used to normalize terms and create new attributes in order to change and expand the original text samples aiming to alleviate factors that can degrade the algorithms performance, such as redundancies and inconsistencies.

Each text sample was normalized and expanded in three different stages, each one generating a new output representation in turn. The stages are:

    Lingo translation: used to translate words in Lingo, which is the name we give to the slangs and abbreviations commonly used on the Internet and SMS, to standard English language.
    Concepts generation: used to obtain all the concepts related to a word, that is, each possible meaning of a certain word.
    Disambiguation: used to find the concept that is more relevant according to the context of the message, among all the concepts related to a certain word.

The Concepts generation and Disambiguation processes are based on the use of the LDB BabelNet, which is a large semantic repository. While the Concepts generation step consists on replacing a given word for all the related concepts, the Disambiguation step automatically selects the most relevant concept for each word. It is performed by means of semantic analysis which detect the context in which the word is found in the sample.

The TextExpansion expands a text sample by splitting it in tokens and processing them in the described stages, generating new expanded samples. This way, given a pre-defined merging rule, the expanded samples are then joined into a final output.

Installation
1. Download the following file: TextExpansion
2. In a Unix-based environment, execute the setup.sh script.

As TextExpansion also uses BabelNet as a semantic dictionary, it can be automatically installed by using the install_babelnet.sh script. However, it is important to advice that it can be time and disk space expensive.

Using TextExpansion
TextExpansion can be used via a graphic UI or via command line. To use its UI just run the textExpansion.py program without any argument. If you want to use it via command line, run the textExpansion.py program passing the arguments on the following format:

$ python textExpansion.py -p <parameters_config_file> -i <input_file> -o <output_file>

where

    <parameters_config_file> is a file indicating which default dictionaries should be used. Basically, such parameter is used to construct the merging rule and each one is an answer (YES/NO) for the following questions: 1. Keep the original terms? 2. Perform lingo translation? 3. Keep the concepts? 4. Perform word sense disambiguation?);
    The next parameter indicates which databases BabelNet should use; it is made up by the answers (YES/NO) to using WordNet and Wikipedia datasets.
    Finally, if you want to use any custom dictionaries, there is a list of its files. Each of these dictionaries should be a text file where each line contains the word that should be translated and its desired translation, separated by a tab. An example of this file can be found on custom_dictionary.txt.
    The parameters configuration file should be in this format (an example file can be found on example.cfg) :

    ORIGINAL = YES/NO
    LINGO = YES/NO
    CONCEPTS = YES/NO
    DISAMBIGUATION = YES/NO

    USE_WORDNET = YES/NO
    USE_WIKIPEDIA = YES/NO

    CUSTOM_DICTIONARIES
    custom_dictionary1.txt
    custom_dictionary2.txt

    <input_file> is the dataset containing the texts you want to normalize and expand. This should be a simple text file with one sample each line. An example of this file can be found on res/samples.txt;
    <output_file> is the file that will contain the expanded dataset. If left empty, the output file will have a default name. 


Publication and More Information
T.P. SILVA, I. SANTOS, T.A. ALMEIDA and J.M. GOMEZ HIDALGO. Normalização Textual e Indexação Semântica Aplicadas na Filtragem de SMS Spam. Anais do XI Encontro Nacional de Inteligência Artificial e Computacional (ENIAC'14), São Carlos, Brazil, October, 2014. (preprint - in portuguese).


About
The TextExpansion has been created by Tiago P. Silva, Tiago A. Almeida, Igor Santos and José María Gómez Hidalgo.

We would like to thank Prof. Roberto Navigli and his team for making the BabelNet available.


(c) Tiago P. Silva, Tiago A. Almeida, Igor Santos and José María Gómez Hidalgo, 2014.
