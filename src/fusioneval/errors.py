class MM8Error(Exception):
    pass

class ConfigError(MM8Error):
    pass

class CheckpointError(MM8Error):
    pass

class ExecutionError(MM8Error):
    pass
