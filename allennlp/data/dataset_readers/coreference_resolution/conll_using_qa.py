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
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer, TokenIndexer
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
        wordpiece_modeling: bool = True,
        max_question_length: int = 64,
        indexer_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._wordpiece_modeling = wordpiece_modeling
        base_indexer_cls = PretrainedTransformerIndexer if self._wordpiece_modeling else PretrainedTransformerMismatchedIndexer
        qa_indexer_cls = PretrainedTransformerIndexer  # have to do matched version for QA
        indexer_kwargs = indexer_kwargs or {}

        self._base_indexer = base_indexer_cls(transformer_model_name, **indexer_kwargs)
        self._base_indexers = {"tokens": self._base_indexer}
        self._qa_indexer = qa_indexer_cls(transformer_model_name, **indexer_kwargs)
        self._qa_indexers = {"tokens": self._qa_indexer}
        self._tokenizer = self._base_indexer._allennlp_tokenizer  # two indexers should have identical tokenizers

        self._max_span_width = max_span_width
        self._max_question_length = max_question_length
        self._max_qa_length = self._qa_indexer._max_length
        assert self._max_question_length < self._max_qa_length

        self._start_of_mention = "<mention>"
        self._end_of_mention = "</mention>"
        self._tokenizer.tokenizer.add_tokens([self._start_of_mention, self._end_of_mention])

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

        if self._wordpiece_modeling:
            flat_sentences_tokens, offsets = self._tokenizer.intra_word_tokenize(
                flattened_sentences
            )
            flattened_sentences = [t.text for t in flat_sentences_tokens]
        else:
            flat_sentences_tokens = [Token(word) for word in flattened_sentences]

        text_field = TextField(flat_sentences_tokens, self._base_indexers)

        cluster_dict: Dict[Tuple[int, int], int] = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id  # type: ignore

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None
        all_questions_with_contexts: List[Field] = []
        all_answers: List[Field] = []

        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(
                sentence, offset=sentence_offset, max_span_width=self._max_span_width
            ):
                orig_start, orig_end = start, end  # we could modify them, so keep a copy

                if self._wordpiece_modeling:
                    start = offsets[start][0]
                    end = offsets[end][1]

                    # `enumerate_spans` uses word-level width limit; here we apply it to wordpieces
                    # We have to do this check here because we use a span width embedding that has
                    # only `self._max_span_width` entries, and since we are doing wordpiece
                    # modeling, the span width embedding operates on wordpiece lengths. So a check
                    # here is necessary or else we wouldn't know how many entries there would be.
                    if end - start + 1 > self._max_span_width:
                        continue
                    # We also don't generate spans that contain special tokens
                    if start < self._tokenizer.num_added_start_tokens:
                        continue
                    if end >= len(flat_sentences_tokens) - self._tokenizer.num_added_end_tokens:
                        continue

                if span_labels is not None:
                    if (orig_start, orig_end) in cluster_dict:
                        span_labels.append(cluster_dict[(orig_start, orig_end)])
                    else:
                        span_labels.append(-1)

                spans.append(SpanField(start, end, text_field))

                questions_with_contexts, answers = self._prepare_qa_input(
                    sentences, orig_start, orig_end, cluster_dict
                )
                all_questions_with_contexts.append(questions_with_contexts)
                all_answers.append(answers)

            sentence_offset += len(sentence)

        span_field = ListField(spans)
        all_questions_with_contexts_field = ListField(all_questions_with_contexts)
        all_answers_field = ListField(all_answers)

        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            if self._wordpiece_modeling:
                for cluster in gold_clusters:
                    for mention_id, mention in enumerate(cluster):
                        start = offsets[mention[0]][0]
                        end = offsets[mention[1]][1]
                        cluster[mention_id] = (start, end)
            metadata["clusters"] = gold_clusters
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "spans": span_field,
            "questions_with_contexts": all_questions_with_contexts_field,
            "answers": all_answers_field,
            "metadata": metadata_field,
        }
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)

    def _prepare_qa_input(
        self, sentences: List[List[str]], curr_span_start: int, curr_span_end: int, cluster_dict: Dict[Tuple[int, int], int]
    ) -> Tuple[ListField[TextField], ListField[SequenceLabelField]]:
        # Find curr sentence
        start_offset = 0
        curr_sentence = None
        for sentence in sentences:
            end = start_offset + len(sentence)
            if start_offset <= curr_span_start <= curr_span_end < end:
                curr_sentence = sentence
                break
            start_offset = end
        assert curr_sentence is not None

        # Find coreferents
        cluster_id = cluster_dict.get((curr_span_start, curr_span_end), -1)
        coreferents = {mention for mention, mention_cluster_id in cluster_dict.items() if mention_cluster_id == cluster_id and mention != (curr_span_start, curr_span_end)}

        rel_span_start = curr_span_start - start_offset
        rel_span_end = curr_span_end - start_offset

        # Prepare question
        question = curr_sentence[:rel_span_start] + [self._start_of_mention] + curr_sentence[rel_span_start:rel_span_end+1] + [self._end_of_mention] + curr_sentence[rel_span_end+1:]
        tokenized_question, _, _ = self._tokenizer.intra_word_tokenize_in_id(question)

        # Strip question from the end until hitting self._end_of_mention, then strip from the start
        n_tokens_to_strip = len(tokenized_question) - self._max_question_length
        n_strip_at_end = min(n_tokens_to_strip, len(tokenized_question) - rel_span_end + 1)
        n_strip_at_start = n_tokens_to_strip - n_strip_at_end
        tokenized_question = tokenized_question[n_strip_at_start:-n_strip_at_end]

        # Prepare context and answers
        context = [word for sentence in sentences for word in sentence]
        pre_context_offset = self._tokenizer.num_added_start_tokens + len(tokenized_question) + self._tokenizer.num_added_middle_tokens
        tokenized_context, context_offsets, _ = self._tokenizer.intra_word_tokenize_in_id(context)
        coreferents = {(context_offsets[start][0], context_offsets[end][1]) for start, end in coreferents}

        n_added_tokens = self._tokenizer.num_added_start_tokens + self._tokenizer.num_added_middle_tokens + self._tokenizer.num_added_end_tokens
        max_context_length = self._max_qa_length - len(tokenized_question) - n_added_tokens

        # Windowing
        qa_text_fields: List[Field] = []
        qa_answers: List[Field] = []
        for i in range(math.ceil(len(context) / max_context_length)):
            segment_start = i * max_context_length
            segment_end = (i + 1) * max_context_length  # exclusive
            context_segment = tokenized_context[segment_start:segment_end]

            tokens = self._tokenizer.ids_to_tokens(tokenized_question, context_segment)
            qa_text_field = TextField(tokens, self._qa_indexers)
            qa_text_fields.append(qa_text_field)

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
            if qa_text_field.sequence_length() != len(answers):
                breakpoint()
            qa_answers.append(SequenceLabelField(answers, qa_text_field))

        return ListField(qa_text_fields), ListField(qa_answers)

    @staticmethod
    def _normalize_word(word):
        if word in ("/.", "/?"):
            return word[1:]
        else:
            return word
