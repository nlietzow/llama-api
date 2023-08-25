import gc
import logging

import tenacity
import torch


def before_sleep(retry_state: tenacity.RetryCallState):
    logging.warning("Cleaning up GPU memory and trying again in 1 second...")
    gc.collect()
    torch.cuda.empty_cache()


retry_on_cuda_oom = tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(1),
    before_sleep=before_sleep,
    retry=tenacity.retry_if_exception_type(torch.cuda.OutOfMemoryError),
)
