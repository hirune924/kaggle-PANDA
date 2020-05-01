#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, List, Dict, Any, Union, Iterable
from pytorch_lightning.logging.neptune import NeptuneLogger
try:
    import neptune
    from neptune.experiments import Experiment
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `neptune` logger which is not installed yet,'  # pragma: no-cover
                      ' install it with `pip install neptune-client`.')
from pytorch_lightning.utilities import rank_zero_only

class MyNeptuneLogger(NeptuneLogger):

    def __init__(self, *args, **kwargs):
        super(MyNeptuneLogger, self).__init__(*args, **kwargs)


    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        try:
            params = self._convert_params(params)
            params = self._flatten_dict(params)
            for key, val in params.items():
                self.experiment.set_property(f'param__{key}', val)
        except:
            neptune.init(api_token=self.api_key,
                         project_qualified_name=self.project_name)
            params = self._convert_params(params)
            params = self._flatten_dict(params)
            for key, val in params.items():
                self.experiment.set_property(f'param__{key}', val)