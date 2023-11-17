from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
    TextSplitter,
)
from typing import Any, Dict, List, Optional, Iterator

# TODO: rewrite this function to use the new TextSplitter for mmd type, and this class is not imported into glue job for now
class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a MarkdownTextSplitter."""
        separators = self.get_separators_for_language(Language.MARKDOWN)
        super().__init__(separators=separators, **kwargs)

def _make_spacy_pipeline_for_splitting(pipeline: str) -> Any:  # avoid importing spacy
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "Spacy is not installed, please install it with `pip install spacy`."
        )
    if pipeline == "sentencizer":
        from spacy.lang.en import English

        sentencizer = English()
        sentencizer.add_pipe("sentencizer")
    else:
        sentencizer = spacy.load(pipeline, exclude=["ner", "tagger"])
    return sentencizer

class NLTKTextSplitter(TextSplitter):
    """Splitting text using NLTK package."""

    def __init__(
        self, separator: str = "\n\n", language: str = "english", **kwargs: Any
    ) -> None:
        """Initialize the NLTK splitter."""
        super().__init__(**kwargs)
        try:
            from nltk.tokenize import sent_tokenize

            self._tokenizer = sent_tokenize
        except ImportError:
            raise ImportError(
                "NLTK is not installed, please install it with `pip install nltk`."
            )
        self._separator = separator
        self._language = language

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = self._tokenizer(text, language=self._language)
        return self._merge_splits(splits, self._separator)

class SpacyTextSplitter(TextSplitter):
    """Splitting text using Spacy package.


    Per default, Spacy's `en_core_web_sm` model is used. For a faster, but
    potentially less accurate splitting, you can use `pipeline='sentencizer'`.
    """

    def __init__(
        self, separator: str = "\n\n", pipeline: str = "en_core_web_sm", **kwargs: Any
    ) -> None:
        """Initialize the spacy text splitter."""
        super().__init__(**kwargs)
        self._tokenizer = _make_spacy_pipeline_for_splitting(pipeline)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits = (s.text for s in self._tokenizer(text).sents)
        return self._merge_splits(splits, self._separator)