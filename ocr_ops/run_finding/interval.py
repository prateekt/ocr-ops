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

    def intersects(self, other: "Interval") -> bool:
        """
        Returns true if the two intervals intersect.
        """
        return self.start < other.end and self.end > other.start

    def overlap_interval(self, other: "Interval") -> "Interval":
        """
        Returns the interval that is the overlap of the two intervals.
        """
        if not self.intersects(other):
            raise ValueError("Intervals do not intersect.")
        return Interval(max(self.start, other.start), min(self.end, other.end))

    def percent_overlap(self, other: "Interval") -> float:
        """
        Returns the percent of this interval that overlaps with the other interval.
        """
        if not self.intersects(other):
            return 0
        overlap = self.overlap_interval(other)
        return len(overlap) / len(self)

    def contains(self, point: int) -> bool:
        """
        Returns true if the interval contains the point.
        """
        return self.start <= point < self.end

    def is_disjoint(self, other: "Interval") -> bool:
        """
        Returns true if the two intervals are disjoint.
        """
        return self.end <= other.start or self.start >= other.end

    def equals(self, other: "Interval") -> bool:
        """
        Returns true if the two intervals are equal.
        """
        return other.start == self.start and other.end == self.end

    def __eq__(self, other: "Interval") -> bool:
        """
        Returns true if the two intervals are equal.
        """
        return self.equals(other)

    def __len__(self) -> int:
        """
        Returns the length of the interval.
        """
        return self.end - self.start

    def __str__(self) -> str:
        """
        Returns the string representation of the interval as start-end.
        """
        return str(self.start) + "-" + str(self.end)
