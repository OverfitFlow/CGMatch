
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .cross_entropy import ce_loss, CELoss
from .consistency import consistency_loss, ConsistencyLoss, GCELoss
from .edl_losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, edl_loss, mse_loss, loglikelihood_loss
from .negative_learning_loss import nl_loss
from .calibration_metrics import calculate_ece, calculate_Brier
