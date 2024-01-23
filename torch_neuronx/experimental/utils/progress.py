def loop_progress(array, label="", logger=None):
    """Similar to tqdm provides a wrapper on the genaerator for a for loop.

    Args:
        array (list): Array of elements to generate from len(array) must return a
                      valid value
        label (str): String prefix for the progress bar
        logger (logging.Logger): Optional logger for output of progress bar
                                 default is std out.

    Yields:
        Any: Next element in the input array

    Examples:
        >>> for i in loop_progress(range(150)):

        >>> array = ['a', 'b', 1, 'sausage']
        >>> for a in loop_progress(array):
    """
    prog = ProgressBar(label=label, logger=logger)

    sz = len(array)
    i = 0

    prog.update(0.0)

    for a in array:
        i += 1
        prog.update(i / sz)
        yield a

    prog.complete()


class ProgressBar:
    def __init__(self, *, label="", width=40, min_interval=None, logger=None):
        """Maintains state for a progress bar and prints the current progress on each call
           to update(fraction). The class assumes that only a fraction
           update greater than min_interval should be displayed

        Args:
            label (str): String prefix for the progress bar
            min_interval (float): Minimum fraction increment to display on update
            logger (logging.Logger): Optional logger for output of progress bar
                                     default is std out.
        """
        self.logger = logger

        if logger is not None:
            self.display = self.log
        else:
            self.display = self.std_out

        self.width = width
        self.label = label
        self.last_fraction = 0.0
        if not min_interval:
            # Make min_interval chunky for logger (20% default) to avoid spamming
            if logger is not None:
                self.min_interval = 0.2
            else:
                self.min_interval = 1.0 / width
        else:
            self.min_interval = min_interval

    def log(self, render_string):
        self.logger.info(render_string)

    def std_out(self, render_string):
        print(render_string, sep="", end="", flush=True)

    def update(self, fraction):
        """Inform the progress bar of a new fraction of progress and display if the
        last change was > min_interval
        """
        left = int(self.width * fraction + 0.5)
        right = self.width - left

        tags = "\u2588" * left
        spaces = " " * right
        percents = f"{100.0*fraction:.1f}%"
        label_out = ""
        if self.label:
            label_out = f"{self.label}: "

        delta_since_last = abs(fraction - self.last_fraction)

        if (fraction < 0.995) and delta_since_last < self.min_interval:
            # Don't render
            return

        render_string = f"\r{label_out}|{tags}{spaces}| {percents} "

        self.display(render_string=render_string)

        self.last_fraction = fraction

    def complete(self):
        self.update(1.0)
        self.display("\n")
