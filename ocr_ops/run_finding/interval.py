class Interval:
    """
    Represents the interval [start, end) in a video.
    """

    def __init__(self, start: int, end: int):
        """
        param start: Start (inclusive) of interval
        param end: End (exclusive) of interval
        """

        if start < 0:
            raise ValueError("Range start must be >= 0.")
        if end < 0:
            raise ValueError("Range end must be >= 0.")
        if start > end:
            raise ValueError("Range start <= end.")
        self.start = start
        self.end = end

    def __eq__(self, other):
        return other.start == self.start and other.end == self.end

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return str(self.start) + "-" + str(self.end)
