import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, Pruner
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import MentionRecall, ConllCorefScores

logger = logging.getLogger(__name__)


@Model.register("coref_using_qa")
class CoreferenceQAResolver(Model):
    """
    This `Model` implements the coreference resolution model described "End-to-end Neural
    Coreference Resolution"
    <https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83>
    by Lee et al., 2017.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    TODO: change

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `BasicTextFieldEmbedder`
        Used to embed the `text` `TextField` we get as input to the model.
    context_layer : `Seq2SeqEncoder`
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : `FeedForward`
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward : `FeedForward`
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size : `int`
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width : `int`
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    lexical_dropout : `int`
        The probability of dropping out dimensions of the embedded text.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str,
        context_layer: Seq2SeqEncoder,
        mention_feedforward: FeedForward,
        antecedent_feedforward: FeedForward,
        feature_size: int,
        max_span_width: int,
        spans_per_word: float,
        max_antecedents: int,
        wordpiece_modeling: bool = True,
        lexical_dropout: float = 0.2,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_embedder = PretrainedTransformerEmbedder(transformer_model_name)
        # Add <mention> and </mention> to vocabulary
        curr_vocab_size = self._text_embedder.transformer_model.config.vocab_size
        self._text_embedder.transformer_model.resize_token_embeddings(curr_vocab_size + 2)
        self._text_embedders = BasicTextFieldEmbedder({"tokens": self._text_embedder})

        self._context_layer = context_layer

        mention_feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)),
        )
        self._mention_pruner = Pruner(mention_feedforward_scorer)

        antecedent_feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(antecedent_feedforward),
            TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1)),
        )
        self._antecedent_pruner = Pruner(antecedent_feedforward_scorer)

        # self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        # self._antecedent_scorer = TimeDistributed(
        #     torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1)
        # )

        self._endpoint_span_extractor = EndpointSpanExtractor(
            context_layer.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False,
        )
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=self._text_embedders.get_output_dim()
        )

        # 10 possible distance buckets.
        self._num_distance_buckets = 10
        self._distance_embedding = Embedding(self._num_distance_buckets, feature_size)

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        text: TextFieldTensors,
        spans: torch.IntTensor,
        metadata: List[Dict[str, Any]],
        span_labels: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        text : `TextFieldTensors`, required.
            The output of a `TextField` representing the text of
            the document.
        spans : `torch.IntTensor`, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a `ListField[SpanField]` of
            indices into the text of the document.
        span_labels : `torch.IntTensor`, optional (default = None).
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.
        metadata : `List[Dict[str, Any]]`, optional (default = None).
            A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
            from this dictionary, which respectively have the original text and the annotated gold coreference
            clusters for that instance.

        # Returns

        An output dictionary consisting of:
        top_spans : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep, 2)` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : `torch.IntTensor`
            A tensor of shape `(num_spans_to_keep, max_antecedents)` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep)` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        device = spans.device
        batch_size = spans.size(0)
        num_spans = spans.size(1)

        num_added_start_tokens = metadata[0]["num_added_start_tokens"]
        num_added_end_tokens = metadata[0]["num_added_end_tokens"]

        # Shape: (batch_size, n_windows, window_length, embedding_size)
        # print(torch.cuda.memory_allocated())
        text_embeddings = self._lexical_dropout(self._text_embedders(text))
        print(text_embeddings.shape)
        # print(torch.cuda.memory_allocated())
        # raise
        # Shape: (batch_size, n_windows, window_length)
        text_mask = util.get_text_field_mask(text).float()

        text_embeddings = text_embeddings[:, :, num_added_start_tokens:-num_added_end_tokens, :]
        text_mask = text_mask[:, :, num_added_start_tokens+num_added_end_tokens:]

        text_embeddings = text_embeddings.reshape(batch_size, -1, text_embeddings.size(-1))
        text_mask = text_mask.reshape(batch_size, -1)

        breakpoint()
        document_length = text_mask.long().sum(dim=1).max()
        text_embeddings = text_embeddings[:, :document_length, :]
        text_mask = text_mask[:, :document_length]

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * document_length))

        (
            _,
            top_span_mask,
            top_span_indices,
            top_span_mention_scores,
        ) = self._mention_pruner(span_embeddings, span_mask, num_spans_to_keep)

        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans, top_span_indices, flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # This huge outer loop seems horrible but since the batch size is often just 1, it's fine
        all_antecedent_indices = []
        all_coref_scores = []
        for i, (batch_top_spans, batch_top_span_mention_scores) in enumerate(zip(top_spans, top_span_mention_scores)):
            prepare_qa_input_function = metadata[i]["prepare_qa_input_function"]
            all_embedded_context = []

            batch_context_length = None
            print(len(batch_top_spans))
            for span in batch_top_spans:
                print('span')
                qa_input_field = prepare_qa_input_function(span, self.vocab)
                print(torch.cuda.memory_allocated())
                qa_input_field = util.move_to_device(qa_input_field, util.get_device_of(spans))
                print(torch.cuda.memory_allocated())
                # (num_windows, window_length, embedding_size)
                embedded_qa_input = self._lexical_dropout(self._text_embedders(qa_input_field))
                print(embedded_qa_input.shape)
                print(torch.cuda.memory_allocated())
                # (num_windows, window_length)
                type_ids = qa_input_field["tokens"]["type_ids"]
                print(torch.cuda.memory_allocated())

                print('-----')

                # (context_length + n_end_tokens, 2)
                context_indices = type_ids.nonzero()
                print(torch.cuda.memory_allocated())
                context_length = context_indices.size(0) - type_ids.size(0) * num_added_end_tokens
                print(torch.cuda.memory_allocated())
                if batch_context_length is not None:
                    assert context_length == batch_context_length
                else:
                    batch_context_length = context_length
                print(torch.cuda.memory_allocated())

                print('======')

                context_start = context_indices[0, 1]
                print(torch.cuda.memory_allocated())
                context_end = context_indices[:, 1].max() - num_added_end_tokens  # inclusive
                print(torch.cuda.memory_allocated())
                window_context_length = context_end - context_start + 1
                print(torch.cuda.memory_allocated())

                print('-=-=-=')

                offset = 0
                while True:
                    offset += window_context_length + num_added_end_tokens
                    if offset >= len(context_indices):
                        break
                    assert context_indices[offset, 1] == context_start
                    assert context_indices[offset - 1, 1] - num_added_end_tokens == context_end
                print(torch.cuda.memory_allocated())

                # (num_windows, window_context_length, embedding_size)
                embedded_context = embedded_qa_input[:, context_start:context_end+1, :]
                print(torch.cuda.memory_allocated())
                # (context_length, embedding_size)
                embedded_context = embedded_context.reshape(-1, embedded_context.size(2))[:context_length, :]
                print(torch.cuda.memory_allocated())
                all_embedded_context.append(embedded_context)
                print(torch.cuda.memory_allocated())

                print('done loop')

            # (num_spans_to_keep, context_length, embedding_size)
            all_embedded_context = torch.stack(all_embedded_context, dim=0)
            # (num_spans_to_keep, num_spans_to_keep, 2 * embedding_size)
            antecedent_embeddings = self._endpoint_span_extractor(all_embedded_context, batch_top_spans)

            (
                _,
                top_antecedent_mask,
                top_antecedent_indices,  # (num_spans_to_keep, max_antecedents)
                top_antecedent_scores,  # (num_spans_to_keep, max_antecedents, 1)
            ) = self._antecedent_pruner(antecedent_embeddings, torch.ones(num_spans_to_keep, num_spans_to_keep, dtype=torch.long, device=device), max_antecedents)
            assert (top_antecedent_mask == 1).all()

            # (num_spans_to_keep, max_antecedents, 1)
            antecedent_mention_scores = util.flattened_index_select(batch_top_span_mention_scores.unsqueeze(0), top_antecedent_indices).squeeze(0)
            coref_scores = (top_antecedent_scores + antecedent_mention_scores).squeeze(-1)
            dummy_scores = coref_scores.new_zeros(coref_scores.size(0), coref_scores.size(1), 1)

            # Shape: (num_spans_to_keep, max_antecedents + 1)
            coref_scores = torch.cat([dummy_scores, coref_scores], -1)
            all_coref_scores.append(coref_scores)
            all_antecedent_indices.append(top_span_indices)

        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = torch.stack(all_coref_scores, dim=0)
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        valid_antecedent_indices = torch.stack(all_antecedent_indices, dim=0)

        # We now have, for each span which survived the pruning stage,
        # a predicted antecedent. This implies a clustering if we group
        # mentions which refer to each other in a chain.
        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = coreference_scores.max(2)
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        predicted_antecedents -= 1

        output_dict = {
            "top_spans": top_spans,
            "antecedent_indices": valid_antecedent_indices,
            "predicted_antecedents": predicted_antecedents,
        }
        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            pruned_gold_labels = util.batched_index_select(
                span_labels.unsqueeze(-1), top_span_indices, flat_top_span_indices
            )

            antecedent_labels = torch.stack([util.flattened_index_select(
                pruned_gold_labels, antecedent_indices
            ).squeeze(-1) for antecedent_indices in valid_antecedent_indices])

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(
                pruned_gold_labels, antecedent_labels
            )
            # Now, compute the loss using the negative marginal log-likelihood.
            # This is equal to the log of the sum of the probabilities of all antecedent predictions
            # that would be consistent with the data, in the sense that we are minimising, for a
            # given span, the negative marginal log likelihood of all antecedents which are in the
            # same gold cluster as the span we are currently considering. Each span i predicts a
            # single antecedent j, but there might be several prior mentions k in the same
            # coreference cluster that would be valid antecedents. Our loss is the sum of the
            # probability assigned to all valid antecedents. This is a valid objective for
            # clustering as we don't mind which antecedent is predicted, so long as they are in
            #  the same coreference cluster.
            top_span_mask = top_span_mask.unsqueeze(-1)
            coreference_log_probs = util.masked_log_softmax(coreference_scores, top_span_mask)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()

            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(
                top_spans, valid_antecedent_indices, predicted_antecedents, metadata
            )

            output_dict["loss"] = negative_marginal_log_likelihood

        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        # Parameters

        output_dict : `Dict[str, torch.Tensor]`, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        # Returns

        The same output dictionary, but with an additional `clusters` key:

        clusters : `List[List[List[Tuple[int, int]]]]`
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
        # the start and end indices of each span.
        batch_top_spans = output_dict["top_spans"].detach().cpu()

        # A tensor of shape (batch_size, num_spans_to_keep) representing, for each span,
        # the index into `antecedent_indices` which specifies the antecedent span. Additionally,
        # the index can be -1, specifying that the span has no predicted antecedent.
        batch_predicted_antecedents = output_dict["predicted_antecedents"].detach().cpu()

        # A tensor of shape (batch_size, num_spans_to_keep, max_antecedents), representing the indices
        # of the predicted antecedents with respect to the 2nd dimension of `batch_top_spans`
        # for each antecedent we considered.
        batch_antecedent_indices = output_dict["antecedent_indices"].detach().cpu()
        batch_clusters: List[List[List[Tuple[int, int]]]] = []

        # Calling zip() on two tensors results in an iterator over their
        # first dimension. This is iterating over instances in the batch.
        for top_spans, antecedent_indices, predicted_antecedents in zip(batch_top_spans, batch_antecedent_indices, batch_predicted_antecedents):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []

            for i, (span, predicted_antecedent) in enumerate(zip(top_spans, predicted_antecedents)):
                if predicted_antecedent < 0:
                    # We don't care about spans which are
                    # not co-referent with anything.
                    continue

                # Find the right cluster to update with this span.
                # To do this, we find the row in `antecedent_indices`
                # corresponding to this span we are considering.
                # The predicted antecedent is then an index into this list
                # of indices, denoting the span from `top_spans` which is the
                # most likely antecedent.
                predicted_index = antecedent_indices[i, predicted_antecedent]

                antecedent_span = (
                    top_spans[predicted_index, 0].item(),
                    top_spans[predicted_index, 1].item(),
                )

                # Check if we've seen the span before.
                if antecedent_span in spans_to_cluster_ids:
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    # We start a new cluster.
                    predicted_cluster_id = len(clusters)
                    # Append a new cluster containing only this span.
                    clusters.append([antecedent_span])
                    # Record the new id of this span.
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

                # Now add the span we are currently considering.
                span_start, span_end = span[0].item(), span[1].item()
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
            batch_clusters.append(clusters)

        output_dict["clusters"] = batch_clusters
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {
            "coref_precision": coref_precision,
            "coref_recall": coref_recall,
            "coref_f1": coref_f1,
            "mention_recall": mention_recall,
        }

    @staticmethod
    def _generate_valid_antecedents(
        num_spans_to_keep: int, max_antecedents: int, device: int
    ) -> Tuple[torch.IntTensor, torch.IntTensor, torch.FloatTensor]:
        """
        This method generates possible antecedents per span which survived the pruning
        stage. This procedure is `generic across the batch`. The reason this is the case is
        that each span in a batch can be coreferent with any previous span, but here we
        are computing the possible `indices` of these spans. So, regardless of the batch,
        the 1st span _cannot_ have any antecedents, because there are none to select from.
        Similarly, each element can only predict previous spans, so this returns a matrix
        of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to
        (i - 1) - j if j <= i, or zero otherwise.

        # Parameters

        num_spans_to_keep : `int`, required.
            The number of spans that were kept while pruning.
        max_antecedents : `int`, required.
            The maximum number of antecedent spans to consider for every span.
        device : `int`, required.
            The CUDA device to use.

        # Returns

        valid_antecedent_indices : `torch.IntTensor`
            The indices of every antecedent to consider with respect to the top k spans.
            Has shape `(num_spans_to_keep, max_antecedents)`.
        valid_antecedent_offsets : `torch.IntTensor`
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            Has shape `(1, max_antecedents)`.
        valid_antecedent_log_mask : `torch.FloatTensor`
            The logged mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            Has shape `(1, num_spans_to_keep, max_antecedents)`.
        """
        # Shape: (num_spans_to_keep, 1)
        target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

        # Shape: (1, max_antecedents)
        valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

        # This is a broadcasted subtraction.
        # Shape: (num_spans_to_keep, max_antecedents)
        raw_antecedent_indices = target_indices - valid_antecedent_offsets

        # In our matrix of indices, the upper triangular part will be negative
        # because the offsets will be > the target indices. We want to mask these,
        # because these are exactly the indices which we don't want to predict, per span.
        # We're generating a logspace mask here because we will eventually create a
        # distribution over these indices, so we need the 0 elements of the mask to be -inf
        # in order to not mess up the normalisation of the distribution.
        # Shape: (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_log_mask = (raw_antecedent_indices >= 0).float().unsqueeze(0).log()

        # Shape: (num_spans_to_keep, max_antecedents)
        valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask

    def _compute_span_pair_embeddings(
        self,
        top_span_embeddings: torch.FloatTensor,
        antecedent_embeddings: torch.FloatTensor,
        antecedent_offsets: torch.FloatTensor,
    ):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        # Parameters

        top_span_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : `torch.IntTensor`, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (1, max_antecedents).

        # Returns

        span_pair_embeddings : `torch.FloatTensor`
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
            util.bucket_values(antecedent_offsets, num_total_buckets=self._num_distance_buckets)
        )

        # Shape: (1, 1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)

        expanded_distance_embeddings_shape = (
            antecedent_embeddings.size(0),
            antecedent_embeddings.size(1),
            antecedent_embeddings.size(2),
            antecedent_distance_embeddings.size(-1),
        )
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(
            *expanded_distance_embeddings_shape
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat(
            [
                target_embeddings,
                antecedent_embeddings,
                antecedent_embeddings * target_embeddings,
                antecedent_distance_embeddings,
            ],
            -1,
        )
        return span_pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(
        top_span_labels: torch.IntTensor, antecedent_labels: torch.IntTensor
    ):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        # Parameters

        top_span_labels : `torch.IntTensor`, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        antecedent_labels : `torch.IntTensor`, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        pairwise_labels_with_dummy_label : `torch.FloatTensor`
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(
        self,
        pairwise_embeddings: torch.FloatTensor,
        top_span_mention_scores: torch.FloatTensor,
        antecedent_mention_scores: torch.FloatTensor,
        antecedent_log_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        # Parameters

        pairwise_embeddings : `torch.FloatTensor`, required.
            Embedding representations of pairs of spans. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, encoding_dim)
        top_span_mention_scores : `torch.FloatTensor`, required.
            Mention scores for every span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_mention_scores : `torch.FloatTensor`, required.
            Mention scores for every antecedent. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_log_mask : `torch.FloatTensor`, required.
            The log of the mask for valid antecedents.

        # Returns

        coreference_scores : `torch.FloatTensor`
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(
            self._antecedent_feedforward(pairwise_embeddings)
        ).squeeze(-1)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = antecedent_scores.new_zeros(*shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores
