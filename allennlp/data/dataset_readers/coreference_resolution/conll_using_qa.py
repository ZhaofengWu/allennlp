import logging
import collections
import math
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    ListField,
    TextField,
    SpanField,
    MetadataField,
    SequenceLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans

logger = logging.getLogger(__name__)

def canonicalize_clusters(
    clusters: DefaultDict[int, List[Tuple[int, int]]]
) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]


@DatasetReader.register("coref_using_qa")
class ConllCorefQAReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format. This dataset reader prepares data for the coref model using
    Transformer QA as described in https://arxiv.org/pdf/1911.01746.pdf.

    Returns a `Dataset` where the `Instances` have four fields : `text`, a `TextField`
    containing the full document text, `spans`, a `ListField[SpanField]` of inclusive start and
    end indices for span candidates, and `metadata`,

    TODO: add here and modify Parameters; wordpiece_modeling is for the coref part only, not for QA

     a `MetadataField` that stores the instance's
    original text. For data with gold cluster labels, we also include the original `clusters`
    (a list of list of index pairs) and a `SequenceLabelField` of cluster ids for every span
    candidate.

    # Parameters

    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = None)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    """

    def __init__(
        self,
        max_span_width: int,
        transformer_model_name: str,
        max_question_length: int = 128,
        max_context_length: int = 384,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._token_indexer = PretrainedTransformerIndexer(transformer_model_name)
        self._token_indexers = {"tokens": self._token_indexer}
        self._tokenizer = self._token_indexer._allennlp_tokenizer

        self._max_span_width = max_span_width
        self._max_question_length = max_question_length
        self._max_context_length = max_context_length

        self._effective_max_context_length = self._max_context_length - self._tokenizer.num_added_start_tokens - self._tokenizer.num_added_end_tokens
        self._effective_max_question_length = self._max_question_length - self._tokenizer.num_added_middle_tokens

        self._start_of_mention = "<mention>"
        self._end_of_mention = "</mention>"
        self._tokenizer.tokenizer.add_tokens([self._start_of_mention, self._end_of_mention])
        self._start_of_mention_id, self._end_of_mention_id = self._tokenizer.tokenizer.convert_tokens_to_ids([self._start_of_mention, self._end_of_mention])

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens, end + total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters = canonicalize_clusters(clusters)
            yield self.text_to_instance([s.words for s in sentences], canonical_clusters)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Instance:

        """
        # Parameters

        sentences : `List[List[str]]`, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : `Optional[List[List[Tuple[int, int]]]]`, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.

        # Returns

        An `Instance` containing the following `Fields`:
            text : `TextField`
                The text of the full document.
            spans : `ListField[SpanField]`
                A ListField containing the spans represented as `SpanFields`
                with respect to the document text.
            span_labels : `SequenceLabelField`, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a `SequenceLabelField`
                 with respect to the `spans `ListField`.
        """
        sentences = [[self._normalize_word(word) for word in sentence] for sentence in sentences]
        flattened_sentences = [word for sentence in sentences for word in sentence]

        tokenized_sentences, offsets = self._intra_word_tokenize_sentences(sentences)
        flattened_tokenized_sentences = [token for sentence in tokenized_sentences for token in sentence]

        windows = self._create_windows(flattened_tokenized_sentences, self._effective_max_context_length)
        text_field = ListField([TextField(self._tokenizer.ids_to_tokens(window), self._token_indexers) for window in windows])

        cluster_dict: Optional[Dict[Tuple[int, int], int]] = None
        if gold_clusters is not None:
            for cluster in gold_clusters:
                for mention_id, mention in enumerate(cluster):
                    start = offsets[mention[0]][0]
                    end = offsets[mention[1]][1]
                    cluster[mention_id] = (start, end)

            cluster_dict = {}
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id  # type: ignore

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        sentence_offset = 0
        for tokenized_sentence in tokenized_sentences:
            for start, end in enumerate_spans(
                tokenized_sentence, offset=sentence_offset, max_span_width=self._max_span_width
            ):
                spans.append(SpanField(start, end))
                if span_labels is not None:
                    if cluster_dict and (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)

            sentence_offset += len(tokenized_sentence)

        span_field = ListField(spans)

        metadata: Dict[str, Any] = {
            "original_text": flattened_sentences,
            "num_added_start_tokens": self._tokenizer.num_added_start_tokens,
            "num_added_middle_tokens": self._tokenizer.num_added_middle_tokens,
            "num_added_end_tokens": self._tokenizer.num_added_end_tokens,
            "prepare_qa_input_function": lambda spans, vocab: self._prepare_qa_input(tokenized_sentences, spans, windows, vocab)
        }
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "spans": span_field,
            "metadata": metadata_field,
        }
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)

    def _intra_word_tokenize_sentences(self, sentences: List[List[str]]):
        sentence_offset = 0
        tokenized_sentences = []
        all_wordpiece_offsets = []
        for sentence in sentences:
            tokenized_sentence, wordpiece_offsets, _ = self._tokenizer.intra_word_tokenize_in_id(sentence, starting_offset=sentence_offset)
            tokenized_sentences.append(tokenized_sentence)
            all_wordpiece_offsets.extend(wordpiece_offsets)
            sentence_offset += len(tokenized_sentence)

        return tokenized_sentences, all_wordpiece_offsets

    def _create_windows(self, ids: List[int], max_length: int) -> List[List[int]]:
        windows = []
        for i in range(math.ceil(len(ids) / max_length)):
            window_start = i * max_length
            window_end = (i + 1) * max_length  # exclusive
            windows.append(ids[window_start:window_end])
        return windows

    def _prepare_qa_input(
        self, tokenized_sentences: List[List[int]], span: Tuple[int, int], context_windows: List[List[int]], vocab
    ) -> TextFieldTensors:
        """
        span: (2,)

        # Returns

        (num_spans, num_segments, max_context_length)
        """
        sentence_offset = 0
        curr_sentence = None
        for sentence in tokenized_sentences:
            if sentence_offset <= span[0] <= span[1] < sentence_offset + len(sentence):
                span = (span[0] - sentence_offset, span[1] - sentence_offset)
                curr_sentence = sentence
                break
            sentence_offset += len(sentence)
        assert curr_sentence is not None

        span_start, span_end = span
        # Prepare question
        question = curr_sentence[:span_start] + [self._start_of_mention_id] + curr_sentence[span_start:span_end+1] + [self._end_of_mention_id] + curr_sentence[span_end+1:]
        # Strip question from the end until hitting self._end_of_mention, then strip from the start
        n_tokens_to_strip = len(question) - self._effective_max_question_length
        n_strip_at_end = min(n_tokens_to_strip, len(question) - span_end + 1)
        n_strip_at_start = n_tokens_to_strip - n_strip_at_end
        question = question[n_strip_at_start:-n_strip_at_end]

        # Prepare context and answers
        questions_with_contexts: List[Field] = []
        for context_window in context_windows:
            tokens = self._tokenizer.ids_to_tokens(question, context_window)
            question_with_context = TextField(tokens, self._token_indexers)
            questions_with_contexts.append(question_with_context)

        input_field = ListField(questions_with_contexts)

        input_field.index(vocab)
        return input_field.as_tensor(input_field.get_padding_lengths())

    @staticmethod
    def _normalize_word(word):
        if word in ("/.", "/?"):
            return word[1:]
        else:
            return word
