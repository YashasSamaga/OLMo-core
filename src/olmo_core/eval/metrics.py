from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import torch
import torch.distributed as dist

from ..distributed.utils import all_reduce_value
from ..utils import get_default_device

__all__ = ["Metric", "MeanMetric", "MeanMetricWithVariance"]


class Metric(metaclass=ABCMeta):
    """
    Base class for evaluation metrics.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.device = device if device is not None else get_default_device()
        self.process_group = process_group

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the metric.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """
        Compute the metric.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the metric.
        """
        raise NotImplementedError

    def as_tensor(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        return value.to(device=self.device, non_blocking=self.device.type != "cpu")


class MeanMetric(Metric):
    """
    Computes the mean over a stream of values.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(device=device, process_group=process_group)
        self.weighted_sum = torch.tensor(0.0, device=self.device)
        self.weight = torch.tensor(0.0, device=self.device)

    def update(
        self, value: Union[float, torch.Tensor], weight: Union[float, torch.Tensor] = 1.0
    ) -> None:
        """
        :param value: The latest value to update the metric with. Could be a tensor of values.
        :param weight: The corresponding weight(s) for the value. Should be the same shape as ``value``.
        """
        value = self.as_tensor(value)
        weight = torch.broadcast_to(self.as_tensor(weight), value.shape)
        if value.numel() == 0:
            return
        self.weighted_sum += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> torch.Tensor:
        """
        Computes the mean over the values and weights given.
        """
        weighted_sum = all_reduce_value(
            self.weighted_sum, device=self.device, group=self.process_group
        )
        weight = all_reduce_value(self.weight, device=self.device, group=self.process_group)
        return weighted_sum / weight

    def reset(self) -> None:
        self.weighted_sum.zero_()
        self.weight.zero_()


class MeanMetricWithVariance(MeanMetric):
    """
    Computes the mean and standard error over a stream of values.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(device=device, process_group=process_group)
        self.weighted_sum_sq = torch.tensor(0.0, device=self.device)

    def update(
        self, value: Union[float, torch.Tensor], weight: Union[float, torch.Tensor] = 1.0
    ) -> None:
        """
        :param value: The latest value to update the metric with. Could be a tensor of values.
        :param weight: The corresponding weight(s) for the value. Should be the same shape as ``value``.
        """
        value = self.as_tensor(value)
        weight = torch.broadcast_to(self.as_tensor(weight), value.shape)
        if value.numel() == 0:
            return
        self.weighted_sum += (value * weight).sum()
        self.weighted_sum_sq += (value.pow(2) * weight).sum()
        self.weight += weight.sum()

    def compute_variance(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the mean, sample variance, and standard error of the mean.
        
        Returns:
            (mean, variance, sem)
        """
        weighted_sum = all_reduce_value(
            self.weighted_sum, device=self.device, group=self.process_group
        )
        weighted_sum_sq = all_reduce_value(
            self.weighted_sum_sq, device=self.device, group=self.process_group
        )
        weight = all_reduce_value(self.weight, device=self.device, group=self.process_group)
        mean = weighted_sum / weight
        variance = (weighted_sum_sq / weight - mean.pow(2)) * (weight / (weight - 1))
        sem = torch.sqrt(variance / weight)
        return mean, variance, sem

    def reset(self) -> None:
        super().reset()
        self.weighted_sum_sq.zero_()
