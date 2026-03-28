"""Dataset adapters."""

from thgkt.data.adapters.assistments import AssistmentsAdapter
from thgkt.data.adapters.ednet import EdNetAdapter
from thgkt.data.adapters.synthetic import SyntheticToyAdapter

__all__ = ["AssistmentsAdapter", "EdNetAdapter", "SyntheticToyAdapter"]
