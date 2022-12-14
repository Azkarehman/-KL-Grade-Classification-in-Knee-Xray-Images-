import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Any, List
from collagen.core import Module


class Session(object):
    """Session class, which implements the basic logic of the training loop.

    Current implementation allows to easily set-up gradient accumulation
    and other strategies.

    Parameters
    ----------
    module : Module
        Instantiated collagen module with trainable parameters.
    optimizer : torch.Optimizer
        Optimizer to train teh model
    loss : torch.nn.Module
        Loss used in the session

    """
    def __init__(self, module: Module, optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module):

        self.__module: Module = module
        self.__optimizer: torch.optim.Optimizer = optimizer
        self.__loss: torch.nn.Module = loss

        # Params of ``backward``
        self.__retain_graph: bool or None = None
        self.__create_graph: bool = False
        self.__gradient = None

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, new_loss: torch.nn.Module):
        self.__loss: torch.nn.Module = new_loss

    def _check_single_element_tuple(self, x):
        return isinstance(x, tuple) and len(x) == 1

    def optimizer_params(self, param_name):
        """Returns the value of optimizer parameter for every group of trainable parameters.
        """
        return [(group['name'], group[param_name]) for group in self.__optimizer.param_groups]

    def set_optimizer_param(self, param_name: str, new_value: Tuple[str, float] or float):
        """Sets a parameter of the optimizer for a particular group of trainable parameters or all groups.

        Parameters
        ----------
        param_name : str
            Name of the optimizer's parameters, e.g. `lr`, `weight_decay` `momentum` etc.
        new_value : Tuple[str, float] or float
            Value of the new parameter. If Tuple, then the first value int specifies the parameters group,
            and the second specifies the actual value.

        """
        for group in self.__optimizer.param_groups:
            if isinstance(new_value, float):
                group[param_name] = new_value[1]
            else:
                if new_value[0] == group['name']:
                    group[param_name] = new_value[1]

    def set_backward_param(self, gradient=None, retain_graph=None, create_graph=False):
        self.__gradient = gradient
        self.__retain_graph = retain_graph
        self.__create_graph = create_graph

    def add_param_group(self, group_name: str):
        """Adds parameter group to the optimizer.

        Parameters
        ----------
        group_name : str
            Name of the group, which needs to be added from model.

        """
        self.__optimizer.add_param_group(self.__module.parameters(group_name))

    def train_step(self, batch: torch.Tensor or Tuple[torch.Tensor],
                   target: torch.Tensor or Tuple[torch.Tensor], retain_graph: bool = False,
                   accumulate_grad: bool = False, return_out=False, with_step: bool = True,
                   callbacks: Tuple[callable] or List[callable] or None = None) -> float:
        """
        Performs one training iteration using the given mini-batch.

        Parameters
        ----------
        batch : torch.Tensor or Tuple[torch.Tensor]
            Mini-batch
        target : torch.Tensor or Tuple[torch.Tensor]
            One or multiple targets
        accumulate_grad : bool
            Whether to zero grad before computing the new gradients.
            False by default, but if True, then the gradients can be accumulated.
            Useful if the batch size are too small because of the input size.
        return_out : bool
            Whether to return output
        callbacks : Tuple[callable] or List[callable] or None
            Callbacks to be used during the training.
        Returns
        -------
        out : float
            Value of the loss

        """

        if not accumulate_grad and self.__optimizer is not None:
            self.__optimizer.zero_grad()

        return self.__batch_step(batch=batch, target=target, with_grad=True,
                                 with_backward=True, with_step=with_step, retain_graph=retain_graph,
                                 return_out=return_out, callbacks=callbacks)

    def eval_step(self, batch: torch.Tensor or Tuple[torch.Tensor],
                  target: torch.Tensor or Tuple[torch.Tensor],
                  return_out=False, retain_graph:bool = False,
                  callbacks: Tuple[callable] or List[callable] or None = None) -> Tuple[float,
                                                                                        torch.Tensor or tuple] or float:
        """
        Performs evaluation of the given mini-batch. If needed, also returns the results.

        Parameters
        ----------
        batch : torch.Tensor or Tuple[torch.Tensor]
            Mini-batch
        target : torch.Tensor or Tuple[torch.Tensor]
            One or multiple targets
        return_out : bool
            Whether to return the output of the network
        callbacks : Tuple[callable] or List [callable] or None
            Callbacks to be used during the training.
        Returns
        -------
        out : Tuple[float, torch.Tensor or tuple] or float
            Result of the evaluation
        """

        return self.__batch_step(batch, target, with_grad=False,
                                 with_backward=False, with_step=False,
                                 eval_mode=True, retain_graph=retain_graph,
                                 return_out=return_out, callbacks=callbacks)

    def __batch_step(self, batch: torch.Tensor or Tuple[torch.Tensor],
                     target: torch.Tensor or Tuple[torch.Tensor] or dict,  with_grad: bool = True,
                     with_backward: bool = True, eval_mode: bool = False, with_step: bool = True,
                     return_out: bool = False, retain_graph: bool = False,
                     callbacks: Tuple[callable] or List[callable] or None = None) -> Tuple[float, Any] or float:
        """
        Private method, which handles the logic for training and evaluation for 1 mini-batch.

        Parameters
        ----------
        batch : torch.Tensor
            Mini-batch
        target : torch.Tensor or Tuple[torch.Tensor]
            One or multiple targets
        with_grad : bool
            Whether to evaluate the given batch with gradient
        with_step : bool
            Whether to evaluate if optimizer step (default: True)
        with_backward : bool
            Whether to perform a backward pass
        eval_mode : bool
            Whether to switch the trained module to the evaluation mode
        return_out : bool
            Whether to return the output
        callbacks : Tuple[callable] or List [callable] or None
            Callbacks to be used during the batch step.

        Returns
        -------
            out : Tuple[float, torch.Tensor or tuple] or float
                Loss value and possibly the output of the model.
        """

        module_device = next(self.__module.parameters()).device
        if callbacks is None:
            callbacks = ()
        if eval_mode:
            with_backward = False
            with_grad = False
            self.__module.train(False)
        else:
            self.__module.train(True)

        if with_backward:
            if not with_grad:
                raise ValueError

        with torch.set_grad_enabled(with_grad):
            if self._check_single_element_tuple(batch):
                batch = batch[0]
            if self._check_single_element_tuple(target):
                target = target[0]

            # Transfer input and target into proper device
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch_on_device = tuple([b.to(module_device) for b in batch])
            elif isinstance(batch, Tensor):
                batch_on_device = batch.to(module_device)
            elif isinstance(batch, dict):
                batch_on_device = dict()
                for k in batch:
                    if isinstance(batch[k], Tensor) and not batch[k].is_cuda:
                        batch_on_device[k] = batch[k].to(module_device)
                    elif isinstance(batch[k], np.ndarray):
                        batch_on_device[k] = torch.tensor(batch[k]).to(module_device)
                    else:
                        batch_on_device[k] = batch[k]
            else:
                raise ValueError('Not support input type {}'.format(type(batch)))

            if isinstance(target, tuple) or isinstance(target, list):
                target_on_device = tuple([t.to(module_device) if isinstance(t, Tensor) else t for t in target])
            elif isinstance(target, dict):
                target_on_device = dict()
                for k in target:
                    if isinstance(target[k], Tensor) and not target[k].is_cuda:
                        target_on_device[k] = target[k].to(module_device)
                    elif isinstance(target[k], np.ndarray):
                        target_on_device[k] = torch.tensor(target[k]).to(module_device)
                    else:
                        target_on_device[k] = target[k]

            elif isinstance(target, Tensor):
                target_on_device = target.to(module_device)
            else:
                raise ValueError('Not support target type {}'.format(type(target)))

            # Forward
            for cb in callbacks:
                cb.on_forward_begin(module=self.__module,
                                    input=batch_on_device,
                                    target=target_on_device,
                                    optimizer=self.__optimizer,
                                    criterion=self.__loss)

            if isinstance(batch_on_device, list) or isinstance(batch_on_device, tuple):
                out = [self.__module(_batch) for _batch in batch_on_device]
            elif isinstance(batch_on_device, dict):
                out = {}
                for k in batch_on_device:
                    if isinstance(batch_on_device[k], Tensor):
                        out[k] = self.__module(batch_on_device[k])
            elif isinstance(batch_on_device, Tensor):
                out = self.__module(batch_on_device)
            else:
                raise ValueError('Not support batch type {}'.format(type(batch_on_device)))

            for cb in callbacks:
                cb.on_forward_end(module=self.__module,
                                  input=batch_on_device,
                                  target=target_on_device,
                                  output=out,
                                  optimizer=self.__optimizer,
                                  criterion=self.__loss)

            # Compute loss
            for cb in callbacks:
                cb.on_loss_begin(session=self,
                                 input=batch_on_device,
                                 target=target_on_device,
                                 output=out)

            loss = self.__loss(out, target_on_device)

            for cb in callbacks:
                cb.on_loss_end(session=self,
                               loss=loss,
                               input=batch_on_device,
                               target=target_on_device,
                               output=out)

            if with_backward:
                # Backward
                for cb in callbacks:
                    cb.on_backward_begin(session=self,
                                         loss=loss,
                                         input=batch_on_device,
                                         target=target_on_device,
                                         output=out)

                loss.backward(gradient=self.__gradient,
                              retain_graph=self.__retain_graph or retain_graph,
                              create_graph=self.__create_graph)

                for cb in callbacks:
                    cb.on_backward_end(session=self,
                                       loss=loss,
                                       input=batch_on_device,
                                       target=target_on_device,
                                       output=out,
                                       optimizer=self.__optimizer,
                                       criterion=self.__loss)

                if with_step:
                    # Optimizer step
                    for cb in callbacks:
                        cb.on_optimizer_step_begin(module=self.__module,
                                                   loss=loss,
                                                   input=batch_on_device,
                                                   target=target_on_device,
                                                   output=out,
                                                   optimizer=self.__optimizer,
                                                   criterion=self.__loss)

                    self.__optimizer.step()

                    for cb in callbacks:
                        cb.on_optimizer_step_end(module=self.__module,
                                                   loss=loss,
                                                   input=batch_on_device,
                                                   target=target_on_device,
                                                   output=out,
                                                   optimizer=self.__optimizer,
                                                   criterion=self.__loss)

            if not return_out:
                return loss.item()
            else:
                return loss.item(), out










