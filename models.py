import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Union, List, Callable

from sentence_transformers import SentenceTransformer, SimilarityFunction
from transformers import (
    PaliGemmaConfig,
    PaliGemmaPreTrainedModel,
    AutoModel,
    AutoModelForCausalLM, Cache, GenerationMixin, StaticCache, HybridCache, PaliGemmaForConditionalGeneration,
    GenerationConfig, LogitsProcessorList, StoppingCriteriaList,
)
from transformers.generation.utils import GenerateOutput
from transformers.utils import logging
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaMultiModalProjector,
)
from helpers import Mode

logger = logging.get_logger(__name__)


class PaliGemmaForClassification(PaliGemmaPreTrainedModel, GenerationMixin):
    def __init__(self, config: PaliGemmaConfig, **kwargs):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        self.config.mode = kwargs.pop('mode', False)
        self.config.num_labels = kwargs.pop('num_labels', 4)
        self.config.attention_pooling = kwargs.pop('attention_pooling', False)

        language_model = AutoModelForCausalLM.from_config(config=config.text_config)

        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        # Custom Classification head + attention layer
        self.output_attention = nn.Linear(config.text_config.hidden_size, 1)
        self.classifier = nn.Linear(config.text_config.hidden_size,
                                    1 if self.config.mode == Mode.SWAG else self.config.num_labels)

        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings with
    # Llava->PaliGemma
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings with
    # Llava->PaliGemma
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings with
    # Llava->PaliGemma
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings with
    # Llava->PaliGemma
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder with
    # Llava->PaliGemma
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder with
    # Llava->PaliGemma
    def get_decoder(self):
        return self.language_model.get_decoder()

    def _update_causal_mask(
            self,
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            input_ids=None,
            inputs_embeds=None,
            is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for
        # prefix is handled below
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )
        return causal_mask

    def get_image_features(self, pixel_values: torch.FloatTensor):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.text_config.hidden_size ** 0.5)
        return image_features

    # ------------------------------
    # Forward for classification
    # ------------------------------
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = True,
            return_dict: Optional[bool] = None,
            num_logits_to_keep: int = 0,
    ):

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, input_ids, inputs_embeds, is_training
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        last_hidden_state = outputs.hidden_states[-1]

        if self.config.attention_pooling:
            # Compute attention scores
            scores = self.output_attention(last_hidden_state).squeeze(-1)

            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))

            # Normalize scores with softmax
            scores = scores - scores.max(dim=-1, keepdim=True).values
            attention_probs = nn.functional.softmax(scores, dim=-1)

            # Compute weighted sum of hidden states
            pooled_output = torch.bmm(attention_probs.unsqueeze(1), last_hidden_state).squeeze(1)
        else:
            pooled_output = last_hidden_state[:, -1, :]

        logits = self.classifier(pooled_output)

        # Pack logits back into their respective multiple-choice bundle corresponding to a single question
        if self.config.mode == Mode.SWAG:
            logits = logits.squeeze(-1)
            batch_size = logits.size(0) // 4
            logits = logits.view(batch_size, 4)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits.to(self.dtype)
            loss = loss_fct(logits, labels)  # labels: shape (batch_size,)

        output = {"loss": loss, "logits": logits}

        return output if return_dict else (loss, logits)


class CosineIndexer:
    def __init__(self):
        self.cosine_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.cosine_model.similarity_fn_name = SimilarityFunction.COSINE

    def convert(self, predictions, options):
        """
        Given a list of string predictions and a list of string options (both in English and Japanese),
        this function returns the index of the option that has the highest cosine similarity with the predicted text.
        """

        indices = []

        for pred, opts in zip(predictions, options):
            # Encode prediction and all options
            text_list = [pred] + opts  # Combine prediction and options
            embeddings = self.cosine_model.encode(text_list, convert_to_numpy=True)

            # Compute cosine similarities between prediction and options
            pred_embedding = embeddings[0].reshape(1, -1)
            option_embeddings = embeddings[1:]

            similarities = self.cosine_model.similarity(pred_embedding, option_embeddings)
            best_index = np.argmax(similarities)

            indices.append(best_index)

        return indices
