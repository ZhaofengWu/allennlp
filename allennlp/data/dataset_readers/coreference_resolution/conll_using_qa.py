import logging
import collections
import math
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union

from overrides import overrides
import torch

from allennlp.common.file_utils import cached_path
from allennlp.data import TextFieldTensors
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
        max_question_length: int = 64,
        indexer_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        indexer_kwargs = indexer_kwargs or {}
        self._token_indexer = PretrainedTransformerIndexer(transformer_model_name, **indexer_kwargs)
        self._token_indexers = {"tokens": self._token_indexer}
        self._tokenizer = self._token_indexer._allennlp_tokenizer

        self._max_span_width = max_span_width
        self._max_question_length = max_question_length
        self._max_qa_length = self._token_indexer._max_length
        assert self._max_question_length < self._max_qa_length

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

        tokenized_sentences, offsets, flat_sentences_tokens = self._intra_word_tokenize_sentences(sentences)
        flattened_sentences = [t.text for t in flat_sentences_tokens]

        text_field = TextField(flat_sentences_tokens, self._token_indexers)

        cluster_dict: Optional[Dict[Tuple[int, int], int]] = None
        if gold_clusters is not None:
            for cluster in gold_clusters:
                for mention_id, mention in enumerate(cluster):
                    start = offsets[mention[0]][0]
                    end = offsets[mention[1]][1]
                    cluster[mention_id] = (start, end)

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
                # We also don't generate spans that contain special tokens
                if start < self._tokenizer.num_added_start_tokens:
                    continue
                if end >= len(flat_sentences_tokens) - self._tokenizer.num_added_end_tokens:
                    continue

                if span_labels is not None:
                    if cluster_dict and (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)

                spans.append(SpanField(start, end, text_field))

            sentence_offset += len(tokenized_sentence)

        span_field = ListField(spans)

        metadata: Dict[str, Any] = {
            "original_text": flattened_sentences,
            "prepare_qa_input_output_function": lambda spans: self._prepare_single_qa_input_output(tokenized_sentences, spans, cluster_dict)
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
        sentence_offset = self._tokenizer.num_added_start_tokens
        tokenized_sentences = []
        all_wordpiece_offsets = []
        for sentence in sentences:
            tokenized_sentence, wordpiece_offsets, _ = self._tokenizer.intra_word_tokenize_in_id(sentence, starting_offset=sentence_offset)
            tokenized_sentences.append(tokenized_sentence)
            all_wordpiece_offsets.extend(wordpiece_offsets)
            sentence_offset += len(tokenized_sentence)

        flat_sentences_tokens = self._tokenizer.ids_to_tokens([word for sentence in tokenized_sentences for word in sentence])
        assert sentence_offset + self._tokenizer.num_added_end_tokens == len(flat_sentences_tokens)

        return tokenized_sentences, all_wordpiece_offsets, flat_sentences_tokens

    def _prepare_qa_input_output(
        self, tokenized_sentences: List[List[int]], spans: torch.LongTensor, cluster_dict: Dict[Tuple[int, int], int] = None
    ) -> TextFieldTensors, torch.LongTensor:
        """
        spans: (batch_size, 2)

        # Returns

        (batch_size, num_segments, max_qa_length)
        (batch_size, num_segments, max_qa_length, num_classes)
        """
        batch_size = spans.size(0)
        device = spans.device
        spans_sentence_index = torch.zeros(batch_size, device=device)
        spans_relative_indices = torch.zeros_like(spans)
        finished_spans = torch.zeros(batch_size, dtype=torch.bool, device=device)

        sentence_offset = self._tokenizer.num_added_start_tokens
        for sentence_idx, sentence in enumerate(tokenized_sentences):
            curr_sentence_spans = (~finished_spans) & (spans - len(sentence) < 0)
            spans_sentence_index[curr_sentence_spans] = sentence_idx
            spans_relative_indices[curr_sentence_spans] -= sentence_offset
            finished_spans[curr_sentence_spans] = 1
            sentence_offset += len(sentence)

        assert finished_spans.all()

        tokenized_context = [token for sentence in tokenized_sentences for token in sentence]

        all_questions_with_contexts: List[Field] = []
        if cluster_dict is not None:
            all_answers: List[Field] = []

        for sentence_idx, span in zip(spans_sentence_index, spans_relative_indices):
            sentence = tokenized_sentences[sentence_idx]
            span = span.tolist()

            # Find coreferents
            coreferents = None
            if cluster_dict is not None:
                cluster_id = cluster_dict.get(span, -1)
                coreferents = [mention for mention, mention_cluster_id in cluster_dict.items() if mention_cluster_id == cluster_id and mention != span]

            result = self._prepare_single_qa_input_output(sentence, tokenized_context, span, coreferents)
            all_questions_with_contexts.append(result["questions_with_contexts"])
            if cluster_dict is not None:
                all_answers.append(result["answers"])

        input_field = ListField(all_questions_with_contexts)
        output_field = ListField(all_answers)

        return input_field.as_tensor(), output_field.as_tensor()

    def _prepare_single_qa_input_output(
        self, curr_sentence: List[int], context: List[int], curr_span: Tuple[int, int], coreferents: List[Tuple[int, int]] = None,
    ) -> Tuple[ListField[TextField], ListField[SequenceLabelField]]:
        curr_span_start, curr_span_end = curr_span
        # Prepare question
        question = curr_sentence[:curr_span_start] + [self._start_of_mention_id] + curr_sentence[curr_span_start:curr_span_end+1] + [self._end_of_mention_id] + curr_sentence[curr_span_end+1:]
        # Strip question from the end until hitting self._end_of_mention, then strip from the start
        n_tokens_to_strip = len(question) - self._max_question_length
        n_strip_at_end = min(n_tokens_to_strip, len(question) - curr_span_end + 1)
        n_strip_at_start = n_tokens_to_strip - n_strip_at_end
        question = question[n_strip_at_start:-n_strip_at_end]

        # Prepare context and answers
        pre_context_offset = self._tokenizer.num_added_start_tokens + len(question) + self._tokenizer.num_added_middle_tokens
        n_added_tokens = self._tokenizer.num_added_start_tokens + self._tokenizer.num_added_middle_tokens + self._tokenizer.num_added_end_tokens
        max_context_length = self._max_qa_length - len(question) - n_added_tokens

        # Windowing
        questions_with_contexts: List[Field] = []
        if coreferents is not None:
            qa_answers: List[Field] = []
        for i in range(math.ceil(len(context) / max_context_length)):
            segment_start = i * max_context_length
            segment_end = (i + 1) * max_context_length  # exclusive
            context_segment = context[segment_start:segment_end]

            tokens = self._tokenizer.ids_to_tokens(question, context_segment)
            question_with_context = TextField(tokens, self._token_indexers)
            questions_with_contexts.append(question_with_context)

            if coreferents is not None:
                # Prepare answers
                answers = ['O'] * len(tokens)
                for start, end in coreferents:
                    # Do not consider out of window ones
                    if not segment_start <= start <= end < segment_end:
                        continue
                    # Adjust indices for (1) window offset and (2) pre context offset
                    start = start - segment_start + pre_context_offset
                    end = end - segment_start + pre_context_offset
                    if answers[start] == 'O':  # Do not overwrite 'I' with 'B'
                        answers[start] = 'B'
                    if end > start:
                        answers[start + 1 : end + 1] = ['I'] * (end - start)

                qa_answers.append(SequenceLabelField(answers, question_with_context))

        output = {"questions_with_contexts": questions_with_contexts}
        if coreferents is not None:
            output["answers"] = qa_answers
        return output

    @staticmethod
    def _normalize_word(word):
        if word in ("/.", "/?"):
            return word[1:]
        else:
            return word
