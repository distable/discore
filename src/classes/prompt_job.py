from .JobArgs import JobArgs


class prompt_job(JobArgs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)