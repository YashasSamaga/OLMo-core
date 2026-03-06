import math
from dataclasses import dataclass

from olmo_core.config import Config


@dataclass
class MuPConfig(Config):
    """
    Configuration for Maximal Update Parameterization (muP).

    When enabled, allows hyperparameters tuned on a small proxy model
    (with ``d_model = base_d_model``) to transfer to the target model.

    The width multiplier ``m = d_model / base_d_model`` determines all scaling factors:

    - Attention: ``softmax_scale = 1/d_head`` instead of ``1/sqrt(d_head)``
    - Output logits: scaled by ``base_d_model / d_model``
    - Output head init: extra ``1/sqrt(m)`` scaling
    - Optimizer LR: hidden/output weights use ``lr * base_d_model / d_model``

    :param base_d_model: The ``d_model`` of the proxy model used for HP tuning.
    """

    base_d_model: int

    def width_mult(self, d_model: int) -> float:
        """
        Width multiplier ``m = d_model / base_d_model``.

        :param d_model: The target model dimensionality.
        """
        return d_model / self.base_d_model

    def lr_scale(self, d_model: int) -> float:
        """
        LR scaling factor for hidden/output weights: ``base_d_model / d_model``.

        :param d_model: The target model dimensionality.
        """
        return self.base_d_model / d_model

    def logit_scale(self, d_model: int) -> float:
        """
        Output logit scaling factor: ``base_d_model / d_model``.

        :param d_model: The target model dimensionality.
        """
        return self.base_d_model / d_model

    def attn_scale(self, head_dim: int) -> float:
        """
        muP attention scaling: ``1/d_head`` instead of ``1/sqrt(d_head)``.

        :param head_dim: The attention head dimension.
        """
        return 1.0 / head_dim

    def output_init_extra_scale(self, d_model: int) -> float:
        """
        Extra scaling for output head init: ``1/sqrt(m)``.

        :param d_model: The target model dimensionality.
        """
        return 1.0 / math.sqrt(self.width_mult(d_model))
