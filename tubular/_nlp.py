from typing import Protocol, List, Optional, Tuple
from tubular.types import DataFrame, Series
import narwhals as nw
from tubular._utils import _convert_dataframe_to_narwhals, _return_narwhals_or_native_dataframe

class ABCLemmatizer(Protocol):

    @classmethod
    def classname(cls) -> str:
        """Method that returns the name of the current class when called."""
        return type(cls).__name__

    def lemmatize(self, text: str)->List[str]:
        ...

    def lemmatize_batch(self, X: nw.DataFrame, column: str)->nw.Series:
        
        backend=nw.get_native_namespace(X)

        lemmas=self.lemmatize(X[column].to_numpy())
        
        return nw.new_series(name=column, values=lemmas, backend=backend)

class MockLemmatizer(ABCLemmatizer):

    def lemmatize(self, text: str)->List[str]:
        return text.split()
    
class SpacyLemmatizer(ABCLemmatizer):

    def __init__(self, model: str="en_core_web_sm", disable: Optional[Tuple[str]]=None):

        try:
            import spacy
        except ImportError as e:
            raise RuntimeError('please install the spacy optional dependency in order to work with SpacyLemmatizer')
        
        self.nlp=spacy.load(model, disable=disable)

        if "lemmatizer" not in self.nlp.pipe_names:
            raise RuntimeError("loaded spacy model has no lemmatizer step in pipeline")
        
    def lemmatize(self, text: str)->List[str]:

        return [t.lemma_ for t in self.nlp(text)]
        