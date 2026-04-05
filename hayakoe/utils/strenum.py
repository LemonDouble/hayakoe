import enum


class StrEnum(str, enum.Enum):
    """
    멤버가 문자열이기도 한(그리고 반드시 문자열이어야 하는) Enum (Python 3.11에서 백포트).
    """

    def __new__(cls, *values: str) -> "StrEnum":
        "values는 이미 `str` 타입이어야 합니다"
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # 반드시 문자열이어야 함
            if not isinstance(values[0], str):  # type: ignore
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # encoding 인수가 문자열인지 확인
            if not isinstance(values[1], str):  # type: ignore
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # errors 인수가 문자열인지 확인
            if not isinstance(values[2], str):  # type: ignore
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        """
        멤버 이름의 소문자 버전을 반환한다.
        """
        return name.lower()
