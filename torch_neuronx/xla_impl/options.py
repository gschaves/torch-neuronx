import warnings


def no_option(*args, **kwargs):
    return None


deprecated_options = []


class OptionsDefault(type):
    def __getattr__(self, item):
        if item in deprecated_options:
            warnings.warn(f"Options.{item} is deprecated. This option will be ignored.")
        else:
            warnings.warn(
                f"Options.{item} does not exist. This option will be ignored."
            )
        return no_option


class Options(metaclass=OptionsDefault):
    pass
