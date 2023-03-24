from transformers import ElectraForPreTraining, ElectraForMultipleChoice, ElectraConfig
from torch import nn
import torch
from transformers.activations import ACT2FN, get_activation

class ElectraDiscriminatorSpanPredictions(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        if hasattr(self, "dense"):
            hidden_states = self.dense(discriminator_hidden_states)
        else:
            hidden_states = self.replaced_dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits

class ElectraForPromptSpanTraining(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)

        self.discriminator_predictions_scratch = ElectraDiscriminatorSpanPredictions(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        label_position=None,
        label_map=None,
        relation_start=None,
        relation_end=None,
    ):

        discriminator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions_scratch(discriminator_sequence_output) # [batch_size,maxlength]


        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits,labels.float())
            loss = torch.sum(loss*label_map)/torch.sum(label_map)
            return loss
        else:
            #prediction mode
            pred = torch.round(logits*label_map)
            return pred






class ElectraDiscriminatorTraining(nn.Module):
    def __init__(self,electra_path):
        super(ElectraDiscriminatorTraining, self).__init__()
        self.config = ElectraConfig.from_pretrained(electra_path)
        self.discriminator = ElectraForPreTraining.from_pretrained(electra_path)

    def forward(self,input_ids, attention_mask=None, token_type_ids=None,labels=None,label_map=None):
        discriminator_outputs = self.discriminator(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        logits = discriminator_outputs[0]
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits,labels.float())
            loss = torch.sum(loss*label_map)/torch.sum(label_map)
            return loss
        else:
            #prediction mode
            pred = torch.round(logits*label_map)
            return pred


class ElectraDiscriminatorTraining_NER(nn.Module):
    def __init__(self,electra_path):
        super(ElectraDiscriminatorTraining_NER, self).__init__()
        self.config = ElectraConfig.from_pretrained(electra_path)
        self.discriminator = ElectraForPreTraining.from_pretrained(electra_path)


    def forward(self,input_ids, attention_mask=None, position_ids=None,labels=None,label_map=None):
        discriminator_outputs = self.discriminator(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids)
        logits = discriminator_outputs[0]
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits,labels.float())
            loss = torch.sum(loss*label_map)/torch.sum(label_map)
            return loss
        else:
            pred = torch.round(logits*label_map)
            return pred