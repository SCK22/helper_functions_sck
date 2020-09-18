import json
import spacy
import en_core_web_md


class TextProcessingClass:
    def __init__(self, text):
        print("Initializing spacy model, this can take some time.")
        nlp = en_core_web_md.load()
        self.text = text
        self.doc = nlp(text)

    def tokenize(self):
        tokens = [token.text for token in self.doc]
        return {"text": self.text, "tokens": tokens}

    def get_pos_tags(self):
        pos_dict = {ent.text: ent.pos_ for ent in self.doc}
        return {"text": self.text, "pos_dict": pos_dict}

    def get_ner_tags(self):
        ner_dict = {ent.text: ent.label_ for ent in self.doc.ents}
        return {"text": self.text, "ner_dict": ner_dict}

    def get_word_vector(self):

        doc_text_vector_exists = {
            ent.text: str(ent.has_vector) for ent in self.doc.ents
        }
        doc_text_vector = {ent.text: str(ent.vector) for ent in self.doc.ents}
        doc_text_vector_norm = {ent.text: str(ent.vector_norm) for ent in self.doc.ents}
        print("doc_text_vector : {}".format(doc_text_vector))
        return {
            "text": self.text,
            "doc_text_vector_exists": doc_text_vector_exists,
            "doc_text_vector": doc_text_vector,
            "doc_text_vector_norm": doc_text_vector_norm,
        }
